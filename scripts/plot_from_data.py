#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import json

def load_run_artifacts(run_dir: str | Path):
    run_dir = Path("predictions") / Path(run_dir)
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

    meta = json.loads(meta_path.read_text())

    splits = SimpleNamespace(X_test=X_test, Y_test=Y_test)
    return splits, pred, meta

def _pick_pred(pred, sample_index: int):
    pred = np.asarray(pred)
    if pred.ndim == 3:
        return pred[sample_index]
    if pred.ndim == 2:
        return pred
    raise ValueError(f"Unsupported prediction shape: {pred.shape}")

def generate_plot_for_feature(
    feature_names,
    splits,
    pred_a,
    pred_b=None,
    label_a="Model A",
    label_b="Model B",
    sample_index: int = 0,
    feature_index: int = 0,
    out_dir: str | Path = "predictions",
) -> str:
    X_test = splits.X_test
    Y_test = splits.Y_test

    if sample_index < 0 or sample_index >= X_test.shape[0]:
        raise IndexError(f"sample_index out of range: {sample_index} (max {X_test.shape[0]-1})")

    if feature_index < 0 or feature_index >= X_test.shape[2]:
        raise IndexError(f"feature_index out of range: {feature_index} (max {X_test.shape[2]-1})")

    input_sequence = X_test[sample_index, :, feature_index]
    true_output = Y_test[sample_index, :, feature_index]

    full_true = np.concatenate([input_sequence, true_output])
    input_len = len(input_sequence)
    out_len = len(true_output)
    full_len = len(full_true)

    time_axis = np.arange(full_len)
    pred_time_axis = np.arange(input_len, input_len + out_len)

    pred_a_2d = _pick_pred(pred_a, sample_index)
    out_a = pred_a_2d[:, feature_index][:out_len]

    out_b = None
    if pred_b is not None:
        pred_b_2d = _pick_pred(pred_b, sample_index)
        out_b = pred_b_2d[:, feature_index][:out_len]

    # 3) plot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 6))
    plt.plot(time_axis, full_true, label="Real Dynamics (Ground Truth)", linewidth=2)
    plt.plot(pred_time_axis, out_a, label=label_a)

    if out_b is not None:
        plt.plot(pred_time_axis, out_b, label=label_b)

    plt.axvline(x=input_len - 1, linestyle=":", label="Fine Input")

    fname = feature_names[feature_index]
    plt.title(f"Comparison Prediction Quantum Dynamics: {fname}")
    plt.xlabel(f"Time Steps (Input: 0-{input_len-1}, Output: {input_len}-{input_len+out_len-1})")
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
    pred_a,
    pred_b=None,
    label_a="Model A",
    label_b="Model B",
    feature_names: list[str] = [],
    sample_index: int = 0,
    out_dir: str = "predictions",
):
    paths = []
    feature_dim = splits.X_test.shape[2]
    for feature_index in range(feature_dim):
        p = generate_plot_for_feature(
            splits=splits,
            pred_a=pred_a,
            pred_b=pred_b,
            label_a=label_a,
            label_b=label_b,
            sample_index=sample_index,
            feature_names=feature_names,
            feature_index=feature_index,
            out_dir=out_dir,
        )
        paths.append(p)
    return paths

# TODO now we plot the value for pred and true value with standardized value but this is not correct for the presentations so we need to perform the denormalization

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate plots from saved raw artifacts (data_splits.npz + predictions.npz). "
                    "Supports single model or comparison between two runs."
    )
    ap.add_argument("--run-a", required=True, help="Path to run directory A (contains data_splits.npz and predictions.npz)")
    ap.add_argument("--run-b", default=None, help="Optional path to run directory B for comparison")
    ap.add_argument("--label-a", default="Model A", help="Legend label for run A")
    ap.add_argument("--label-b", default="Model B", help="Legend label for run B")
    ap.add_argument("--sample", type=int, default=0, help="Sample index in X_test/Y_test")
    ap.add_argument("--feature", type=int, default=None, help="If set, plot only this feature index (0-based)")
    ap.add_argument("--out-dir", default=None, help="Output directory for plots. Default: <run-a>/plots or compare dir")
    return ap.parse_args()

def main():
    args = parse_args()

    splits_a, pred_a, meta_a = load_run_artifacts(args.run_a)

    pred_b = None
    if args.run_b is not None:
        splits_b, pred_b, meta_b = load_run_artifacts(args.run_b)

        # safety check: to compare, X_test and Y_test should match
        if splits_a.X_test.shape != splits_b.X_test.shape or splits_a.Y_test.shape != splits_b.Y_test.shape:
            raise ValueError(
                "Runs A and B have different X_test/Y_test shapes. "
                f"A: X{splits_a.X_test.shape}, Y{splits_a.Y_test.shape} | "
                f"B: X{splits_b.X_test.shape}, Y{splits_b.Y_test.shape}"
            )

    # choose output dir
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        if args.run_b is None:
            out_dir = Path("predictions/" + args.run_a) / "plots"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("predictions") / "compare" / f"compare__{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.feature is not None:
        p = generate_plot_for_feature(
            splits=splits_a,
            pred_a=pred_a,
            pred_b=pred_b,
            label_a=args.label_a,
            label_b=args.label_b,
            feature_names= meta_a.get("feature_names", []),
            sample_index=args.sample,
            feature_index=args.feature,
            out_dir=str(out_dir),
        )
        print(f"Saved: {p}")
    else:
        paths = generate_all_plots(
            splits=splits_a,
            pred_a=pred_a,
            pred_b=pred_b,
            label_a=args.label_a,
            label_b=args.label_b,
            feature_names= meta_a.get("feature_names", []),
            sample_index=args.sample,
            out_dir=str(out_dir),
        )
        print(f"Saved {len(paths)} plots in: {out_dir}")


if __name__ == "__main__":
    main()
