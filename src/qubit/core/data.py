from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, Tuple, List
from ..utils.seed import set_seed
from ..dataclasses.dataset_splits import DatasetSplits
import numpy as np
import pandas as pd


from .standardizer import fit_standardizer, apply_standardizer

from ..dataclasses.sr_config import SuperResolutionConfig
from ..dataclasses.data_config import DataConfig

def compute_feature_dim(n: int) -> int:
   # number of features: n magnetisations (10) + (n * (n - 1)) // 2 correlations (45)
    return n + (n * (n - 1)) // 2


def load_raw_dataframe(csv: str) -> pd.DataFrame:

    csv_path = Path(csv).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # header = None because our csv file doesn't have the header in the first row
    return pd.read_csv(csv_path, header=None)

def randomize_traj_to_use(n_traj, num_traj_to_use, time_steps):
    rows = []

    traj_selected = np.random.choice(n_traj, size=num_traj_to_use, replace=False).tolist()
    # traj_selected = np.sort(np.random.choice(n_traj, size=num_traj_to_use, replace=False)).tolist()
    for i in traj_selected:
        rows += list(range(i * time_steps, i * time_steps + time_steps))
    return rows


def build_trajectories(df: pd.DataFrame, data_cfg : DataConfig) -> Tuple[np.ndarray, list[str]]:

    # n_traj = 400 => number of trajectories 
    # timesteps every 0.02 second * 1001 (n_timesteps x each trajectory) = 20 seconds 

    # number of rows => timesteps * n_traj  => 1001 * 400 => 400.400

    # n => number of used qubit => in the csv there are total_qubits = 10 
    # number of cols => 1 (cols of time_steps) + feature_dim => 1 + (10 (magnetisations) + 45 (correlations) ) = 56 (for n=10)

    used_qubits = data_cfg.dataset.used_qubits  # number of qubits to use
    traj_fraction = data_cfg.dataset.traj_fraction # fraction of trajectories to use

    # fixed values 
    total_qubits = data_cfg.dataset.total_qubits # number of qubits in the csv
    time_steps = data_cfg.dataset.time_steps # number of time steps per trajectory
    n_traj = data_cfg.dataset.n_traj # number of trajectories
    corr_start = 1 + total_qubits # start from the 12th element (11 index)

    feature_dim = compute_feature_dim(used_qubits)
    
    # used only a small amount of trajectories for training
    num_traj_to_use = int(n_traj * traj_fraction)

    # at least we want to use one trajectory
    if num_traj_to_use <= 0:
        raise ValueError("traj_fraction too small, no trajectories to use")

    # magnetization cols: m1..mk 
    mag_cols = list(range(1, 1 + used_qubits))

    corr_cols = correlation_columns(corr_start=corr_start, used_qubits=used_qubits, total_qubits= total_qubits)

    # total cols to extract => list of index to extract from the dataframe
    cols = mag_cols + corr_cols

    # (rows, feature_dim)

    # df.shape => (400400, 56) => (n_traj * time_steps, 1 + feature_dim)
    
    # randomly select num_traj_to_use trajectories
    rows = randomize_traj_to_use(n_traj,num_traj_to_use,time_steps)

    # df.iloc[rows_selector, cols_selector] => extract all rows and only the selected cols
    data_features = df.iloc[rows, cols].to_numpy(dtype=np.float32)
    # data_features.shape => (len(rows), len(cols)) 

    # convert data_features first into 1d (shape become data_features.shape[0] * data_features.shape[1]) and reshape it into 3d
    data_3d = data_features.reshape(num_traj_to_use, time_steps, feature_dim)

    # data_3d.shape => (num_traj_to_use, time_steps, feature_dim)

    # feature names
    feat_names = [f"m({i})" for i in range(1, used_qubits + 1)]
    feat_names += [f"c({i},{j})" for i in range(1, used_qubits) for j in range(i + 1, used_qubits + 1)]

    # print(feat_names)

    return data_3d, feat_names

def correlation_columns(corr_start: int, used_qubits: int, total_qubits: int) -> list[int]:
    
    # c12 c13 c14 c15 c16 c17 c18 c19 c1,10
    # c23 c24 c25 c26 c27 c28 c29 c2,10
    # c34 c35 c36 c37 c38 c39 c3,10
    # c45 c46 c47 c48 c49 c4,10
    # ...

    corr_cols = []
    size = used_qubits - 1
    stride = total_qubits - 1

    while size > 0:
        corr_cols_temp = list(range(corr_start, corr_start + size))
        corr_cols = corr_cols + corr_cols_temp
        corr_start += stride
        size -= 1
        stride -= 1
        
    return corr_cols

def prepare_dataset(data_cfg: DataConfig, sr_cfg: SuperResolutionConfig | None 
 ) -> Tuple[DatasetSplits, list[str], np.ndarray, np.ndarray]:

    seed = data_cfg.split.seed
    set_seed(seed, deterministic=True)
    
    # take in input the csv path and return the dataframe
    df = load_raw_dataframe(data_cfg.dataset.csv_path)

    traj_3d, feat_names = build_trajectories(df, data_cfg)
    # feature_dim => magnetizations + correlations
    # traj_3d.shape => (num_traj_to_use, time_steps, feature_dim)

    # input_len: piece of sequence that we give in input to the model
    input_len = data_cfg.windowing.input_seq_len

    # stride for the sliding window
    stride = data_cfg.windowing.stride

    # percentages for validation and test sets
    val_ratio = data_cfg.split.val_ratio
    test_ratio = data_cfg.split.test_ratio
    
    # output_len is a piece of sequence that we want the model to predict
    output_len = data_cfg.windowing.output_seq_len
    

    splits, mean, std = split_by_trajectory_then_window(
        traj_3d,
        input_len=input_len,
        output_len=output_len,
        stride=stride,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        sr_cfg=sr_cfg,
    )
    
    return splits, feat_names, mean, std

def split_by_trajectory_then_window(
    traj_3d: np.ndarray,   # (n_traj, time_steps, feature_dim)
    input_len: int,
    output_len: int,
    stride: int,
    val_ratio: float, # greater than 0 and less or equal than 1 
    test_ratio: float,  # greater than 0 and less or equal than 1 
    sr_cfg : SuperResolutionConfig | None
) -> tuple[DatasetSplits, np.ndarray, np.ndarray]:

    # from now on the n_traj is the number of trajectories after the fractioning
    
    # n_traj => number of trajectories 

    # * n_traj are already shuffled in prepare_dataset function 
    n_traj = traj_3d.shape[0]

    # number of trajectories for each split 
    n_test = int(round(n_traj * test_ratio))
    n_val = int(round(n_traj * val_ratio))

    # list of indices for each split
    test_idx = list(range(0,n_test))
    val_idx = list(range(n_test,n_test + n_val))
    train_idx = list(range(n_test + n_val,n_traj))

    # split trajectories
    tr_train = traj_3d[train_idx]
    tr_val = traj_3d[val_idx]
    tr_test = traj_3d[test_idx]

    # standardize based on training set 
    mean, std = fit_standardizer(tr_train)

    # mean.shape == (1, 1, feature_dim)
    # std.shape == (1, 1, feature_dim)
    
    # apply standardization using mean and std from training set 
    tr_train = apply_standardizer(tr_train, mean, std)
    tr_val = apply_standardizer(tr_val, mean, std)
    tr_test = apply_standardizer(tr_test, mean, std)
    # X - mean / std

    # windowing after splitting to prevent data leakage
    if sr_cfg is not None : 
        X_train, Y_train = make_sr_windows_from_trajectories(tr_train, input_len,stride, sr_cfg)
        X_val, Y_val = make_sr_windows_from_trajectories(tr_val, input_len,stride, sr_cfg)
        X_test, Y_test = make_sr_windows_from_trajectories(tr_test, input_len,stride, sr_cfg)
    else : 
        X_train, Y_train = make_windows_from_trajectories(tr_train, input_len, output_len, stride)
        X_val, Y_val = make_windows_from_trajectories(tr_val, input_len, output_len, stride)
        X_test, Y_test = make_windows_from_trajectories(tr_test, input_len, output_len, stride)

    # *     X_* and Y_*
    # *     X_*.shape[0] from now on don't have num_traj but num_windows
    # *     num_windows is calculated as n_traj * number of windows for each trajectory
    # *     X_*.shape => (num_windows , time_steps , feature_dim) 
    
    return DatasetSplits(
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
    ), mean,std

def make_windows_from_trajectories(
    traj_3d: np.ndarray,  # (n_traj_split, time_steps, feature_dim)
    input_len: int,
    output_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
   
    # time_steps => length of each trajectory => 1001
    T = traj_3d.shape[1]

    # total window size 
    win_size = input_len + output_len

    if win_size > T:
        raise ValueError(f"time_steps={T} too small for input+output={win_size}")


    X_list, Y_list = [], []

    # for each trajectory, extract windows with given stride

    for tr in traj_3d:
        # tr.shape => (time_steps, feature_dim)
        # s => start index of each window
        for s in range(0, T - win_size + 1, stride):
            # each element of this list is a 2d array
            X_list.append(tr[s:s+input_len, :])
            Y_list.append(tr[s+input_len:s+win_size, :])
            
    # len(X_list) => each element is a windows
    # we want an 3d array so use stack to convert each element that contain 2d array
    # X.shape(n_windows = len(X_list) , timesteps = X_list[i].shape[0], feature_dim = X_list[i].shape[1] )

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    #print(len(X_list)) == print(X.shape[0])

    # X => inputs : (n_windows, input_len, feature_dim)
    # Y => targets: (n_windows, output_len, feature_dim)
    return X, Y

def make_sr_windows_from_trajectories(
    traj_3d: np.ndarray, # (n_traj_split, time_steps, feature_dim)
    input_seq_len: int, # (low-res length)
    slide_stride: int, # data.windowing.stride
    sr_cfg : SuperResolutionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    
    window_len = input_seq_len * sr_cfg.stride

    T = traj_3d.shape[1] # timesteps 
    
    # X_list shape(num_windows, window_len, feature_dim + 1 (mask channel)) => inputs model with time holes filled with mask_value 
    # the additional feature_dim is the mask_channel useful for the model to understand the difference between the observed and missed timestep s
    
    # Y_list (num_windows , window_len, feature_dim) => ground truth all the timesteps either observed and also not observed (without mask_value applied)
    X_list, Y_list = [], []

    # list of index from 0 to windows_len - 1
    idx = np.arange(window_len, dtype=np.int32)  

    # create a boolean mask 1d when the index is observed put True else False (missed) and
    # after we convert it in float mask with 0.0 or 1.0 values 
    obs_1d = ((idx - sr_cfg.offset) % sr_cfg.stride == 0).astype(np.float32) 
    # obs_1d.shape(windows_len)

    # obs_1d[i] == 1 is the specific index == timestep is observed 
    # obs_1d[i] == 0 is the specific index == timestep is missed (hole) 

    obs = obs_1d[:, None] 
    # obs.shape(winows_len, None)

    # traj_3d.shape(n_traj_split, time_steps, feature_dim)
    for traj in traj_3d:

        # traj.shape(time_steps, feature_dim)

        # here we apply the windowing stride 
        for start in range(0, T - window_len + 1, slide_stride):
            
            # here work in single window

            y = traj[start:start + window_len, :] # (L,F) ground truth
            # y.shape(window_len, feature_dim)

            x = y.copy()
            # miss boolean array .shape(windows_len)

            # miss => is the opposite of obs_1d
            # => True where the index is missed 
            # => False where the index is observed 
            miss = (obs_1d == 0)

            # if the index of miss[i] == True so x[i,:] = mask_value (for all feature_dim)
            x[miss, :] = sr_cfg.mask_value

            # before: x.shape(window_len, feature_dim) and obs.shape(windows_len,None)
            # after: x_in.shape(window_len, feature_dim + 1)
            # if last feature is == 1 the specific timestep t is observed (not a hole)
            # if last feature is == 0 the specific timestep t is a hole (missed)
            x_in = np.concatenate([x, obs], axis=-1) 

            X_list.append(x_in)
            Y_list.append(y)

    # X.shape(num_windows, windows_len, feature_dim + 1)
    # Y.shape(num_windows, windows_len, feature_dim)
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y