from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List
from .seed import set_seed
from ..model.dataset_splits import DatasetSplits
import numpy as np
import pandas as pd



from ..model.data_config import DataConfig

def compute_feature_dim(n: int) -> int:
   # number of features: n magnetisations (10) + (n * (n - 1)) // 2 correlations (45)
    return n + (n * (n - 1)) // 2


def load_raw_dataframe(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")
    return pd.read_csv(csv_path, header=None)



def build_trajectories(df: pd.DataFrame, data_cfg : DataConfig) -> Tuple[np.ndarray, list[str]]:

    # number of rows => n_points * n_traj => 400.400
    # number of cols => 1 + feature_dim => 1 + (10 (magnetisations) + 45 (correlations) ) = 56 (for n=10)

    total_qubits = data_cfg.dataset.total_qubits # number of qubits
    used_qubits = data_cfg.dataset.used_qubits  # number of qubits to use
    time_steps = data_cfg.dataset.time_steps # number of time steps per trajectory
    n_traj = data_cfg.dataset.n_traj # number of trajectories

    traj_fraction = data_cfg.dataset.traj_fraction # fraction of trajectories to use


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
    feat_names = [f"m({i})" for i in range(1, used_qubits + 1)]
    feat_names += [f"c({i},{j})" for i in range(1, used_qubits) for j in range(i + 1, used_qubits + 1)]


    # #TODO : instead of to take the first needed rows we could randomize the trajectories to use with a seed 

    # # take only the needed rows and reshape to 3d array (num_trajectories, time_steps, feature_dim)
    # data_3d = data_features[:needed].reshape(num_traj_to_use, time_steps, feature_dim)

    #TODO : implement the npz saving the standarzed dataset or simple dataset so that we can load directly the npz file without preprocessing again
    
    print(feat_names)
    # split into input (X) and output (Y) sequences

    return data_3d, feat_names


# def load_or_prepare_dataset(m: Dict[str, Any]) -> Tuple[DatasetSplits, list[str]]:

#     # given a string  path of csv data, result => pandaas dataframe
#     df = load_raw_dataframe(m["dataset"]["csv_path"])

#     data_3d , feat_names = build_seq2seq(df, m)
    
#     # take the split parameters
#     seed = int(m["split"]["seed"])

#     # set for all libraries the seed
#     set_seed(seed, deterministic=True)
    
#     # take the value of splitting
#     val_ratio = float(m["split"]["val_ratio"])
#     test_ratio = float(m["split"]["test_ratio"])

#     splits = split_by_trajectory(X, Y, val_ratio=val_ratio, test_ratio=test_ratio)
#     return splits, feat_names

def prepare_dataset(data_cfg : DataConfig ) -> Tuple[DatasetSplits, list[str]]:

    df = load_raw_dataframe(data_cfg.dataset.csv_path)

    traj_3d, feat_names = build_trajectories(df, data_cfg)

    seed = data_cfg.split.seed
    set_seed(seed, deterministic=True)

    input_len  =  data_cfg.windowing.input_seq_len
    output_len =  data_cfg.windowing.output_seq_len
    stride     =  data_cfg.windowing.stride

    val_ratio  = data_cfg.split.val_ratio
    test_ratio = data_cfg.split.test_ratio

    splits = split_by_trajectory_then_window(
        traj_3d,
        input_len=input_len,
        output_len=output_len,
        stride=stride,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    return splits, feat_names

def make_windows_from_trajectories(
    traj_3d: np.ndarray,  # (n_traj_split, time_steps, feature_dim)
    input_len: int,
    output_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    T = traj_3d.shape[1]

    win = input_len + output_len

    if T < win:
        raise ValueError(f"time_steps={T} too small for input+output={win}")

    X_list, Y_list = [], []

    # for each trajectory, extract windows with given stride
    for tr in traj_3d:
        for s in range(0, T - win + 1, stride):

            X_list.append(tr[s:s+input_len, :])
            Y_list.append(tr[s+input_len:s+win, :])

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


def split_by_trajectory_then_window(
    traj_3d: np.ndarray,  
    input_len: int,
    output_len: int,
    stride: int,
    val_ratio: float,
    test_ratio: float,
) -> DatasetSplits:
    
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and (val_ratio + test_ratio) < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0,1) and val_ratio + test_ratio < 1")

    # n => number of trajectories 
    n = traj_3d.shape[0]

    idx = np.random.permutation(n)

    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    # split trajectories
    tr_train = traj_3d[train_idx]
    tr_val   = traj_3d[val_idx]
    tr_test  = traj_3d[test_idx]

    # standardize based on training set 
    mean, std = fit_standardizer(tr_train)

    tr_train = apply_standardizer(tr_train, mean, std)
    tr_val   = apply_standardizer(tr_val, mean, std)
    tr_test  = apply_standardizer(tr_test, mean, std)

    # windowing after splitting to prevent data leakage
    X_train, Y_train = make_windows_from_trajectories(tr_train, input_len, output_len, stride)
    X_val,   Y_val   = make_windows_from_trajectories(tr_val,   input_len, output_len, stride)
    X_test,  Y_test  = make_windows_from_trajectories(tr_test,  input_len, output_len, stride)

    return DatasetSplits(
        X_train=X_train, Y_train=Y_train,
        X_val=X_val,     Y_val=Y_val,
        X_test=X_test,   Y_test=Y_test,
    )



def fit_standardizer(X_train: np.ndarray, eps: float = 1e-8):

    mean = X_train.mean(axis=(0, 1), keepdims=True)  
    std  = X_train.std(axis=(0, 1), keepdims=True) + eps
    return mean, std

def apply_standardizer(A: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (A - mean) / std
