from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List
from .seed import set_seed
from ..model.dataset_splits import DatasetSplits
import numpy as np
import pandas as pd




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

    total_qubits = int(cfg["dataset"]["total_qubits"]) # number of qubits
    used_qubits = int(cfg["dataset"]["used_qubits"]) # number of qubits to use
    time_steps = int(cfg["dataset"]["time_steps"]) # number of time steps per trajectory
    n_traj = int(cfg["dataset"]["n_traj"]) # number of trajectories

    traj_fraction = float(cfg["dataset"]["traj_fraction"]) # fraction of trajectories to use

    input_len = int(cfg["windowing"]["input_seq_len"]) # number of input time steps for neural network
    output_len = int(cfg["windowing"]["output_seq_len"]) # number of time steps to predict for neural network

    feature_dim = compute_feature_dim(used_qubits)
    num_corr = feature_dim - used_qubits

    # used only a small amount of trajectories for training
    num_traj_to_use = max(1, int(n_traj * traj_fraction))

    # number of trajectories we want to use * number of time steps per trajectory
    needed_rows = num_traj_to_use * time_steps

    # magnetization cols: m1..mk 
    mag_cols = list(range(1, 1 + used_qubits))

    # correlation cols: c[1+total_qubits] : c[(1+total_qubits) + num_corr]
    corr_start = 1 + total_qubits
    corr_cols = list(range(corr_start, corr_start + num_corr))

    # total cols to extract
    cols = mag_cols + corr_cols

    # (rows, feature_dim)
    data_features = df.iloc[:, cols].to_numpy(dtype=np.float32) 

    # safety check
    if data_features.shape[0] < needed_rows:
        raise ValueError(f"Insufficient rows: {data_features.shape[0]} < {needed_rows}")

    # reshape diretto
    data_3d = data_features[:needed_rows].reshape(num_traj_to_use, time_steps, feature_dim)

    # nomi feature
    feat_names = [f"m{i}" for i in range(1, used_qubits + 1)]
    feat_names += [f"c{i}{j}" for i in range(1, used_qubits) for j in range(i + 1, used_qubits + 1)]


    # #TODO : instead of to take the first needed rows we could randomize the trajectories to use with a seed 

    # # take only the needed rows and reshape to 3d array (num_trajectories, time_steps, feature_dim)
    # data_3d = data_features[:needed].reshape(num_traj_to_use, time_steps, feature_dim)

    print(feat_names)
    # split into input (X) and output (Y) sequences
    X = data_3d[:, :input_len, :]
    Y = data_3d[:, input_len: input_len + output_len, :]
    return X, Y


def load_or_prepare_dataset(m: Dict[str, Any]) -> DatasetSplits:

    # given a string  path of csv data, result => pandaas dataframe
    df = load_raw_dataframe(m["dataset"]["csv_path"])

    X, Y = build_seq2seq(df, m)
    
    # take the split parameters
    seed = int(m["split"]["seed"])

    # set for all libraries the seed
    set_seed(seed, deterministic=True)
    
    # take the value of splitting
    val_ratio = float(m["split"]["val_ratio"])
    test_ratio = float(m["split"]["test_ratio"])

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

