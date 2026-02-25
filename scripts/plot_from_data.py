import argparse
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import json
import re

from qubit.utils.config_values import PREDICTION_PATH
from qubit.core.standardizer import inverse_standardizer 

# datatime 
RUN_RE = re.compile(r"_(\d{8})_(\d{6})$") 

def find_latest_run_dir() -> str:
    base_dir = Path("runs/" + PREDICTION_PATH)

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    best = None  # (datetime, Path)

    # for each sub_dir that contain the base_dir 
    for p in base_dir.rglob("*"):
        if not p.is_dir(): continue
        
        # TODO when we change the name of dir and remove the datatime this function don't work
        # filter the dir that not contain the datatime
        m = RUN_RE.search(p.name)
        if not m: continue
        
        # extract date+hour and convert in datetime 
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        
        # takes the most recent dir
        if best is None or dt > best[0]:
            best = (dt, p)

    if best is None:
        raise ValueError(f"No directory with pattern *_YYYYMMDD_HHMMSS found inside {base_dir}")
    
    # return the path of the more recent dir
    return str(best[1])

def load_run_artifacts(run_str: str, destandardize: bool):

    # run_dir = Path("runs" / "predictions") / ...
    run_dir = Path(run_str)
    
    data_path = run_dir / "data_splits.npz"
    pred_path = run_dir / "predictions.npz"
    meta_path = run_dir / "meta.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    data = np.load(data_path)
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    pred_npz = np.load(pred_path, allow_pickle=True)
    pred = pred_npz["pred"]

    mean = data["mean"] if "mean" in data.files else None
    std = data["std"]  if "std" in data.files else None

    
    if destandardize:
        if mean is None or std is None:
            raise ValueError("destandardize=True but mean/std not found in data_splits.npz")

        feature_dim = mean.shape[-1]

        for i in range(Y_test.shape[0]): print(Y_test[i,0,0],end=" ")
        print()

        Y_test = inverse_standardizer(Y_test, mean, std)
        for i in range(Y_test.shape[0]): print(Y_test[i,0,0],end=" ")
        print()
        pred = inverse_standardizer(pred, mean, std)

        if X_test.shape[-1] == feature_dim:
            X_test = inverse_standardizer(X_test, mean, std)

        elif X_test.shape[-1] == feature_dim + 1:
            X_feat = inverse_standardizer(X_test[:, :, :feature_dim], mean, std)
            X_mask = X_test[:, :, feature_dim:]  # keep mask unchanged
            X_test = np.concatenate([X_feat, X_mask], axis=-1)

        else:
            raise ValueError(
                f"Unexpected X_test last dim {X_test.shape[-1]} "
                f"(expected {feature_dim} or {feature_dim+1})"
            )

    meta = json.loads(meta_path.read_text())

    splits = SimpleNamespace(X_test=X_test, Y_test=Y_test)
    return splits, pred, meta

def generate_plot_for_feature(
        feature_names,
        splits,
        pred,
        sample_index,
        feature_index,
        label="Model",
        out_dir: str | Path = PREDICTION_PATH,
    ):
    
    X_test = splits.X_test
    Y_test = splits.Y_test

    # FORECASTING 
    # X_test.shape(num_windows, input_seq_len, feature_dim)
    # Y_test.shape(num_windows, output_seq_len, feature_dim)

    # SUPER RESOLUTION 
    # X_test.shape(num_windows, window_size, feature_dim + 1 (mask channel))
    # Y_test.shape(num_windows, window_size, feature_dim)

    # ex. sample_index is in [0,2,4]
    if sample_index < 0 or sample_index >= X_test.shape[0]:
        raise IndexError(f"sample_index out of range: {sample_index} (max {X_test.shape[0]-1})")

    if feature_index < 0 or feature_index >= Y_test.shape[2]:
        raise IndexError(f"feature_index out of range: {feature_index} (max {Y_test.shape[2]-1})")

    # with sample_index we plot only a specific index window and a specific feature_index
    x_feat = X_test[sample_index, :, feature_index]
    # If Forecasting => x_feat.shape(input_seq_len)

    # If SuperResolution => x_feat.shape(window_size)

    y_true = Y_test[sample_index, :, feature_index]
    # If Forecasting => y_true.shape(output_seq_len)
    # If SuperResolution => y_true.shape(window_size)

    # decide mode
    is_sr = (X_test.shape[2] == Y_test.shape[2]+1)

    out_dir = Path(str(out_dir) + f"/s({sample_index})")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 6))
    fname = feature_names[feature_index]

    if not is_sr:
        # --- FORECASTING ---

        # the next index of input_sequence[-1] is true_output[0] 

        # the full ground truth are the array with size = input_sequence + true_output
        full_true = np.concatenate([x_feat, y_true])
        input_len = len(x_feat)
        out_len = len(y_true)
        full_len = len(full_true)

        # return an numpy array with values from 0 to full_seq-1
        time_axis = np.arange(full_len)
        
        out = pred[sample_index, :, feature_index]
        if out.shape[0] != out_len:
            raise ValueError(f"predlength {out.shape[0]} != out_len {out_len}")
    
        pred_time_axis = np.arange(input_len, input_len + out_len)

        plt.plot(time_axis, full_true, label="Ground Truth", linewidth=2)
        plt.plot(pred_time_axis, out, label=label)

        plt.axvline(x=input_len - 1, linestyle=":", label="End of Input")
        plt.xlabel(f"Time (Input 0-{input_len-1}, Output {input_len}-{input_len+out_len-1})")

    else:
        # --- SUPER RESOLUTION ---
        L = Y_test.shape[1]
        t = np.arange(L)

        out = pred[sample_index, :, feature_index] # (L,)
        if out.shape[0] != L:
            raise ValueError(f"pred length {out.shape[0]} != window {L}")

        # mask channel: 1 = observed, 0 = missing (to predict)
        mask = X_test[sample_index, :, -1] # (L,)
        miss_idx = np.where(mask < 0.5)[0]

        # Nice, readable palette (matplotlib "tab" colors)
        c_gt = "tab:gray" # ground truth line
        c_obs = "tab:green" # observed input points
        c = "tab:blue" # model predicted points
        c_err = "tab:red" # error connectors

        # Ground truth as a line (neutral color so it doesn't compete with predictions)
        plt.plot(t, y_true, color=c_gt, label="Ground Truth", linewidth=2.2, zorder=1)

        # Observed input points (only where mask=1)
        plt.scatter(
            miss_idx, y_true[miss_idx],
            s=22, color=c_obs, edgecolors="white", linewidths=0.5,
            label="Holes (mask=1)", zorder=3
        )

        # Predicted points ONLY where mask=0 (missing)
        plt.scatter(
            miss_idx, out[miss_idx],
            s=26, color=c, edgecolors="white", linewidths=0.6,
            label=f"{label} preds (mask=0)", zorder=4
        )

        # Error connectors (GT - prediction) for missing indices
        for j, i in enumerate(miss_idx):
            plt.plot(
                [i, i], [y_true[i], out[i]],
                color=c_err, linewidth=1.0, alpha=0.75,
                label="Residual (GT-pred)" if j == 0 else None,
                zorder=2
            )

        plt.xlabel("Time (window)")
        plt.title(f"Super-Resolution: {fname}")

    plt.title(f"Comparison Prediction Quantum Dynamics: {fname}")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True)

    filename = f"s({sample_index})_f({feature_index+1})_{fname}.jpg"
    plot_path = out_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(plot_path)

def generate_all_plots(
    splits,
    pred,
    meta,
    label="Model",
    out_dir: str = PREDICTION_PATH,
):
    paths = []
    
    # ! we take the feature_dim from Y because in SR X_test.shape[2] == feature_dim + 1 (mask channel)
    feature_dim = splits.Y_test.shape[2]
    for s in meta.get("sample_index"):
        for feature_index in range(feature_dim):
            p = generate_plot_for_feature(
                splits=splits,
                pred=pred,
                label=label,
                sample_index=s,
                feature_names = meta.get("feature_names", []),
                feature_index=feature_index,
                out_dir=out_dir
            )
            paths.append(p)
    return paths


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate plots from saved raw artifacts (data_splits.npz + predictions.npz). "
                    "Supports single model or comparison between two runs."
    )
    ap.add_argument("--run", default=None, help="Path to run directory (contains data_splits.npz and predictions.npz)")
    ap.add_argument("--label", default="Model", help="Legend label for run")
    ap.add_argument("--feature", type=int, default=None, help="If set, plot only this feature index (0-based)")
    ap.add_argument("--out-dir", default=None, help="Output directory for plots. Default: <run-a>/plots")
    return ap.parse_args()

def main():
    args = parse_args()
    
    # ! important must indicate if the X and Y are standardized or not 
    plot_in_original_units = True

    if args.run is None: args.run = find_latest_run_dir()

    splits, pred, meta = load_run_artifacts(args.run,plot_in_original_units)


    # choose output dir
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.run) / "prediction_plots"

    
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.feature is not None:
        p = generate_plot_for_feature(
            splits=splits,
            pred=pred,
            label=args.label,
            feature_names= meta.get("feature_names", []),
            sample_index=meta.get("sample_index"),
            feature_index=args.feature,
            out_dir=str(out_dir),
        )
        print(f"Saved: {p}")
    else:
        paths = generate_all_plots(
            splits=splits,
            pred=pred,
            label=args.label,
            meta = meta,
            out_dir=str(out_dir),
        )
        print(f"Saved {len(paths)} plots in: {out_dir}")


if __name__ == "__main__":
    main()

