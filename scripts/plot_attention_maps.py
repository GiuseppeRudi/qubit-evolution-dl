import argparse
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt

from qubit.utils.config_values import PREDICTION_PATH

RUN_RE = re.compile(r"_(\d{8})_(\d{6})$")

def find_latest_run_dir() -> str:
    base_dir = Path("runs/" + PREDICTION_PATH)
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    best = None  # (datetime, Path)
    for p in base_dir.rglob("*"):
        if not p.is_dir():
            continue
        m = RUN_RE.search(p.name)
        if not m:
            continue
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        if best is None or dt > best[0]:
            best = (dt, p)

    if best is None:
        raise ValueError(f"No directory with pattern *_YYYYMMDD_HHMMSS found inside {base_dir}")
    return str(best[1])

def _to_2d_heat(attn_4d: np.ndarray, *, sample: int , head: int) -> np.ndarray:
    """
    attn_4d: (B, H, Tq, Tk)
    head = -1 -> average heads
    """
    if attn_4d.ndim != 4:
        raise ValueError(f"Expected attn with 4 dims (B,H,Tq,Tk), got {attn_4d.shape}")

    B, H, _, _ = attn_4d.shape

    if sample < 0 or sample >= B:
        raise IndexError(f"sample index out of range: {sample} (B={B})")

    a = attn_4d[sample]  # (H,Tq,Tk)
    
    if head == -1:
        return a.mean(axis=0)  # (Tq,Tk)
    if head < 0 or head >= H:
        raise IndexError(f"head out of range: {head} (H={H})")
    return a[head]  # (Tq,Tk)


def _downsample(mat: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return mat
    return mat[::stride, ::stride]

def plot_heatmap(
    mat2d: np.ndarray,
    *,
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.imshow(mat2d, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description="Plot attention maps from attn_maps.npz saved in a run dir.")

    ap.add_argument("--run", default=None, help="Run directory. If omitted, uses latest run under runs/<PREDICTION_PATH>.")

    ap.add_argument("--attn-file", default="attn_maps.npz", help="Attention npz filename inside run dir.")
    
    ap.add_argument("--out-dir", default=None, help="Output directory. Default: <run>/attn_plots")

    ap.add_argument("--sample", type=int, default=0, help="Batch index to plot (default 0).")

    ap.add_argument("--head", type=int, default=-1, help="Which head to plot. -1 means average over heads.")

    return ap.parse_args()

def plot_one(key: str, z, out_dir, args):
    arr = z[key]  # (B,H,Tq,Tk)
    mat = _to_2d_heat(arr, sample = args.sample, head = args.head)

    # automatic downsample 
    # Tq, Tk = mat.shape
    ds = 1
    # if max(Tq, Tk) >= 800: ds = 8
    # elif max(Tq, Tk) >= 300: ds = 4
    # elif max(Tq, Tk) >= 200: ds = 2

    mat = _downsample(mat, ds)

    # Labels depending on type
    if key.startswith("enc/") and key.endswith("/self"):
        xlabel, ylabel = "Key timestep (Tin)", "Query timestep (Tin)"
    elif key.startswith("dec/") and key.endswith("/self"):
        xlabel, ylabel = "Key timestep (Tout)", "Query timestep (Tout)"
    elif key.startswith("dec/") and key.endswith("/cross"):
        xlabel, ylabel = "Encoder timestep (Tin)", "Decoder timestep (Tout)"
    else:
        xlabel, ylabel = "Key", "Query"

    head_str = "avg_heads" if args.head == -1 else f"head{args.head}"
    ds_str = f"ds{ds}" 
    safe_key = key.replace("/", "_")
    fname = f"{safe_key}__{head_str}__{ds_str}.jpg"

    title = f"{key} | sample={args.sample} | {head_str} | {ds_str}"
    plot_heatmap(mat, title=title, out_path=out_dir / fname, xlabel=xlabel, ylabel=ylabel)
    print(f"Saved: {out_dir / fname}")

def main():
    args = parse_args()

    run_dir = Path(args.run) if args.run is not None else Path(find_latest_run_dir())
    attn_path = run_dir / args.attn_file

    if not attn_path.exists():
        raise FileNotFoundError(f"Missing attention file: {attn_path}")

    out_dir = Path(run_dir / "attn_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(attn_path, allow_pickle=True)

    for k in z.files: plot_one(k, z, out_dir, args)

if __name__ == "__main__":
    main()
