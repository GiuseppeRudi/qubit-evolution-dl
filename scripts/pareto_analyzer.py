from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Literal
import numpy as np
import pandas as pd

def is_pareto_efficient(costs: np.ndarray, maximize_mask: np.ndarray | None = None) -> np.ndarray:

    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if is_efficient[i]:
            if maximize_mask is None:
                dominates = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
            else:
                better_or_equal = np.where(maximize_mask, costs >= costs[i], costs <= costs[i])
                strictly_better = np.where(maximize_mask, costs > costs[i], costs < costs[i])
                dominates = np.all(better_or_equal, axis=1) & np.any(strictly_better, axis=1)

            is_efficient[i] = not np.any(dominates & (np.arange(n_points) != i))

    return is_efficient


def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    costs = df[["values_score", "values_out_seq_len"]].to_numpy(dtype=float)
    maximize_mask = np.array([False, True])
    df["is_pareto"] = is_pareto_efficient(costs, maximize_mask)
    return df

Criteria = Literal["best_score", "max_output_len", "knee"]

def select_best_by_criteria(
    pareto_df: pd.DataFrame,
    criteria: Criteria,
) -> pd.Series[Any]:

    if pareto_df.empty:
        raise ValueError("No Pareto-optimal points found!")

    score_col = "values_score"
    out_len_col = "values_out_seq_len"

    df = pareto_df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[out_len_col] = pd.to_numeric(df[out_len_col], errors="coerce")
    df = df.dropna(subset=[score_col, out_len_col])

    if df.empty:
        raise ValueError(
            f"Pareto DataFrame has no valid numeric rows for columns "
            f"'{score_col}' and '{out_len_col}'."
        )

    if criteria == "best_score":
        return df.nsmallest(1, score_col).iloc[0]

    if criteria == "max_output_len":
        return df.nlargest(1, out_len_col).iloc[0]

    if criteria == "knee":
        pareto = df.copy()

        scores = pareto[score_col].to_numpy(dtype=float)
        lengths = pareto[out_len_col].to_numpy(dtype=float)

        s_rng = scores.max() - scores.min()
        l_rng = lengths.max() - lengths.min()
        if s_rng == 0 or l_rng == 0:
            return pareto.iloc[len(pareto)//2]

        s = (scores - scores.min()) / s_rng         
        l = (lengths - lengths.min()) / l_rng        
        # distance to ideal (0,1)
        d = np.sqrt(s**2 + (1.0 - l)**2)
        knee_idx = int(np.argmin(d))
        return pareto.iloc[knee_idx]
    
    raise ValueError(f"Unknown criteria: {criteria}")

def create_simple_pareto_plot(df: pd.DataFrame, best_trial: pd.Series | None = None,
                              save_path: str | None = None, show: bool = True):
    fig, ax = plt.subplots(figsize=(10, 7))

    non_pareto = df[~df['is_pareto']]
    pareto = df[df['is_pareto']].copy()

    ax.scatter(non_pareto['values_score'], non_pareto['values_out_seq_len'], 
              c='lightgray', s=100, alpha=0.6, label='Dominated Trials', 
              edgecolors='gray', linewidth=0.5, zorder=1)

    pareto_sorted = pareto.sort_values('values_out_seq_len')
    ax.plot(pareto_sorted['values_score'], pareto_sorted['values_out_seq_len'], 
            'b-', linewidth=2, alpha=0.7, zorder=2)
    ax.scatter(pareto['values_score'], pareto['values_out_seq_len'], 
              c='red', s=200, marker='*', label='Pareto Front', # type: ignore
              edgecolors='darkred', linewidth=1.5, zorder=5)

    if best_trial is not None:
        x = float(best_trial["values_score"])
        y = float(best_trial["values_out_seq_len"])
        n = int(best_trial["number"])

        ax.scatter(
            x, y,
            c="gold", s=260, marker="*", # type: ignore
            edgecolors="black", linewidth=1.2,
            zorder=10, label="Selected (knee)"
        )

        ax.annotate(
            f"Knee point\nTrial #{n}",
            xy=(x, y),
            xytext=(30, 25),              
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="black",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.9),
            zorder=11,
        )

    for _, row in pareto.iterrows():
        ax.annotate(f"#{int(row['number'])}", 
                   (row['values_score'], row['values_out_seq_len']),
                   textcoords="offset points", xytext=(8, 5), 
                   fontsize=9, fontweight='bold')

    ax.set_xlabel('Score (lower is better)', fontsize=12)
    ax.set_ylabel('Output Sequence Length (higher is better)', fontsize=12)
    ax.set_title('Pareto Front Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show: plt.show()
    else: plt.close()

def main():

    parser = argparse.ArgumentParser(
        description='Analyze Pareto front from Optuna Level 2 tuning results',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('csv_file', type=str, help='Path to the Optuna report CSV file')
    parser.add_argument('--criteria', type=str, default='knee',
                       choices=['best_score', 'max_output_len', 'knee'],
                       help='Criteria for selecting the best trial (default: knee)')

    parser.add_argument('--plot', type=str, default='pareto_plot.jpg', 
                       help='Generate plot (default: pareto_plot.jpg if no filename given)')
    
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot (only save)')

    args = parser.parse_args()

    # Load data
    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)

    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} complete trials from {args.csv_file}")

    df.columns = df.columns.str.strip()  

    df["values_score"] = pd.to_numeric(df["values_score"].astype(str).str.strip(), errors="coerce")
    df["values_out_seq_len"] = pd.to_numeric(df["values_out_seq_len"].astype(str).str.strip(), errors="coerce")

    df = df.dropna(subset=["values_score", "values_out_seq_len"])

    # Validate required columns
    required_cols = ['values_score', 'values_out_seq_len', 'number']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)

    # Filter complete trials only
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
        df = df[df["state"] == "COMPLETE"].copy()


    # Compute Pareto front
    df = compute_pareto_front(df)
    pareto_df = df[df['is_pareto']].copy()

    print(f"Found {len(pareto_df)} Pareto-optimal trials")

    best_trial = select_best_by_criteria(pareto_df, args.criteria)

    # Generate plot if requested
    if args.plot:
        plot_path = args.plot 
        show_plot = not args.no_show
        print(f"Selected best trial based on criteria '{args.criteria}': Trial #{int(best_trial['number'])} with score {best_trial['values_score']} and output length {best_trial['values_out_seq_len']}")
        create_simple_pareto_plot(df, best_trial, save_path=plot_path, show=show_plot)


if __name__ == '__main__':
    main()

"""
# Full analysis with 4-panel plot
python pareto_analyzer.py report.csv --criteria best_score --plot analysis.png

# Simple plot for presentations
python pareto_analyzer.py report.csv --criteria knee --plot-type simple --plot simple.png

# Balanced selection with visualization
python pareto_analyzer.py report.csv --criteria balanced --weight 0.3 --plot balanced.png

# Export and plot without displaying
python pareto_analyzer.py report.csv --criteria max_output_len --plot output.png --no-show --export params.json
"""