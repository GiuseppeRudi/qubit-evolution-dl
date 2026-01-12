from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

phase_colors = {
    "TeacherForcingPhase":"#FFDDDD",
    "MaskedModelingPhase": "#DDFFDD",
    "ScheduledSamplingPhase": "#DDDDFF",
    "FullAutoregressivePhase": "#FFFFDD"
}

phase_labels = {
    "TeacherForcingPhase": "Teacher Forcing",
    "MaskedModelingPhase": "Masked Modeling",
    "ScheduledSamplingPhase": "Scheduled Sampling",
    "FullAutoregressivePhase": "Full Autoregressive"
}

def save_loss_plots_keras(
    run_dir: str | Path,
    history,
    phases,
    fr_key: str,   
    train_key: str = "loss",
    val_key: str = "val_loss",
    prefix: str = "loss",
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """
    Save 4 plot:
      - {prefix}_train.jpg  : train_key (es. loss)
      - {prefix}_val.jpg    : val_key (es. val_loss)
      - {prefix}_fr.jpg     : fr_key (es. test_fr_loss)
      - {prefix}_all.jpg    : combined (train + val + fr)

    Returns: (train_path, val_path, fr_path, all_path)
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(history, "history") or not isinstance(history.history, dict):
        raise TypeError("Expected a Keras History object (returned by model.fit), with a .history dict")

    h = history.history

    train = h.get(train_key, None)
    val = h.get(val_key, None)
    fr = h.get(fr_key, None)

    train_path: Path | None = None
    val_path: Path | None = None
    fr_path: Path | None = None
    all_path: Path | None = None

    epoch_start = 0
    phase_intervals = []
    for ph in phases:
        start = epoch_start
        end = epoch_start + ph.epochs
        phase_intervals.append((start, end, type(ph).__name__))
        epoch_start = end

    # --- 1) Train plot ---
    if train is not None and len(train) > 0:
        y = np.asarray(train, dtype=float)
        x = np.arange(1, len(y) + 1)

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Epoch")
        plt.ylabel(train_key)
        plt.title(f"Training {train_key}")
        plt.grid(True)

        train_path = run_dir / f"{prefix}_train.jpg"
        plt.savefig(train_path, dpi=300, bbox_inches="tight")
        plt.close()

    # --- 2) Val plot ---
    if val is not None and len(val) > 0:
        y = np.asarray(val, dtype=float)
        x = np.arange(1, len(y) + 1)

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Epoch")
        plt.ylabel(val_key)
        plt.title(f"Validation {val_key}")
        plt.grid(True)

        val_path = run_dir / f"{prefix}_val.jpg"
        plt.savefig(val_path, dpi=300, bbox_inches="tight")
        plt.close()

    # --- 3) Free-running plot ---
    if fr is not None and any(v is not None for v in fr):

        x = [epoch + 1 for epoch, v in enumerate(fr) if v is not None]
        y = [v for v in fr if v is not None]

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Epoch")
        plt.ylabel(fr_key)
        plt.title(f"Free-running {fr_key}")
        plt.grid(True)

        fr_path = run_dir / f"{prefix}_fr.jpg"
        plt.savefig(fr_path, dpi=300, bbox_inches="tight")
        plt.close()

    # --- 4) Combined plot  ---
    has_any = (
        (train is not None and len(train) > 0) or
        (val is not None and len(val) > 0) or
        (fr is not None and any(v is not None for v in fr))
    )
    if has_any:
        plt.figure(figsize=(8,5))

        # sfondi per fase
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start + 0.5, end + 0.5, color=phase_colors[ph_type], alpha=0.2)

        # plot train/val/fr
        if train is not None and len(train) > 0:
            y = np.asarray(train, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=f"train ({train_key})")

        if val is not None and len(val) > 0:
            y = np.asarray(val, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=f"val ({val_key})")

        if fr is not None and any(v is not None for v in fr):
            x = [epoch + 1 for epoch, v in enumerate(fr) if v is not None]
            y = [v for v in fr if v is not None]
            plt.plot(x, y, label=f"free-running ({fr_key})")

        # linee verticali per inizio fase
        for start, _, _ in phase_intervals:
            plt.axvline(start + 0.5, color='k', linestyle='--', linewidth=0.8)

        # legenda per fasi
        legend_patches = [mpatches.Patch(color=phase_colors[ph_type], alpha=0.2, label=phase_labels[ph_type])
                        for _, _, ph_type in phase_intervals]
        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_patches)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss curves (TF + Free-running)")
        plt.grid(True)

        all_path = run_dir / f"{prefix}_all.jpg"
        plt.savefig(all_path, dpi=300, bbox_inches="tight")
        plt.close()

    return train_path, val_path, fr_path, all_path
