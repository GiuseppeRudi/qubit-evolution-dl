from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from .seed import set_seed
from .config_loader import load_dataset_config
from ..model.dataset_splits import DatasetSplits

import numpy as np
import pandas as pd

from .config_loader import get_project_root
from .config_loader import load_yaml



def compute_feature_dim(n: int) -> int:
   # number of features: n magnetisations (10) + (n * (n - 1)) // 2 correlations (45)
    return n + (n * (n - 1)) // 2


def load_raw_dataframe(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")
    return pd.read_csv(csv_path, header=None)





def build_seq2seq(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:

    # number of rows => n_points * n_traj => 400.400
    # number of cols => 1 + feature_dim => 1 + (10 (magnetisations) + 45 (correlations) ) = 56 (for n=10)
    
    n = int(cfg["dataset"]["n"]) # number of qubits
    time_steps = int(cfg["dataset"]["time_steps"]) # number of time steps per trajectory
    n_traj = int(cfg["dataset"]["n_traj"]) # number of trajectories

    traj_fraction = float(cfg["dataset"]["traj_fraction"]) # fraction of trajectories to use

    input_len = int(cfg["windowing"]["input_seq_len"]) # number of input time steps for neural network
    output_len = int(cfg["windowing"]["output_seq_len"]) # number of time steps to predict for neural network

    feature_dim = compute_feature_dim(n) # magnetisations + correlations
    total_columns = 1 + feature_dim # time column + features

    # only for safety check
    # .shape => (rows,cols)  => df.shape[0] => take the rows and .shape[1] => take the columns
    if df.shape[1] != total_columns:
        print(f" Expected {total_columns} cols, obtained {df.shape[1]}.")

    # take all rows and all columns except the first (time) => convert to numpy array of type float32
    data_features = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    # used only a small amount of trajectories for training
    num_traj_to_use = max(1, int(n_traj * traj_fraction))

    # number of trajectories we want to use * number of time steps per trajectory
    needed = num_traj_to_use * time_steps
    
    # only for safety check
    if data_features.shape[0] < needed:
        raise ValueError(f"Insufficient rows: {data_features.shape[0]} < {needed}")

    # take only the needed rows and reshape to 3d array (num_trajectories, time_steps, feature_dim)
    data_3d = data_features[:needed].reshape(num_traj_to_use, time_steps, feature_dim)

    # only for safety check
    max_out = time_steps - input_len
    out_len = min(output_len, max_out)

    # split into input (X) and output (Y) sequences
    X = data_3d[:, :input_len, :]
    Y = data_3d[:, input_len:input_len + out_len, :]
    return X, Y


def load_or_prepare_dataset(cfg_path: Path | str) -> DatasetSplits:

    # given a configurations file path , result => dictionary with dataset config
    cfg = load_dataset_config(cfg_path)

    # given a file path of csv data, result => pandaas dataframe
    df = load_raw_dataframe(cfg["dataset"]["csv_path"])


    X, Y = build_seq2seq(df, cfg)

    # take the split parameters
    seed = int(cfg["split"]["seed"])

    # set for all libraries the seed
    set_seed(seed, deterministic=True)
    
    # take the value of splitting
    val_ratio = float(cfg["split"]["val_ratio"])
    test_ratio = float(cfg["split"]["test_ratio"])


    splits = split_by_trajectory(X, Y, val_ratio=val_ratio, test_ratio=test_ratio)
    return splits




def split_by_trajectory(
    X: np.ndarray,
    Y: np.ndarray,
    val_ratio: float,
    test_ratio: float, ) -> DatasetSplits:

    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and (val_ratio + test_ratio) < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0,1) and val_ratio + test_ratio < 1")

    # number of used trajectories 
    n = X.shape[0]
    
    # important for seed randomization
    idx = np.random.permutation(n)

    # number of trajectories for test and validation
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    # boundary conditions 
    if n_test + n_val >= n:
        n_test = min(n_test, n - 2) if n >= 2 else 0
        n_val = min(n_val, n - 1 - n_test) if n - n_test >= 1 else 0

    # generate a list of indexes to trajector
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return DatasetSplits(
        X_train=X[train_idx],
        Y_train=Y[train_idx],
        X_val=X[val_idx],
        Y_val=Y[val_idx],
        X_test=X[test_idx],
        Y_test=Y[test_idx],
    )
