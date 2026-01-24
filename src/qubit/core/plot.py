from pathlib import Path
import re
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ..dataclasses.training_config import PhaseConfig
from ..dataclasses.training_config import TrainingConfig

phase_colors = {
    "TeacherForcingPhase": "red",
    "MaskedModelingPhase": "green",
    "ScheduledSamplingPhase": "blue",
    "FullAutoregressivePhase": "yellow"
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
    training_cfg: TrainingConfig,
    fr_key: str,
    output_seq_len: int,
    val_key: str = "val_",
    prefix: str = "loss",
) -> tuple[Path | None, Path | None, Path | None, Path | None]:

    run_dir = Path(str(run_dir) + "/loss_plots")
    run_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(history, "history") or not isinstance(history.history, dict):
        raise TypeError("Expected a Keras History object (returned by model.fit), with a .history dict")

    h = history.history
    print(h)

    target = fr_key + "target_" + prefix + "_" + str(output_seq_len)
    phase = fr_key + "phase_" + prefix
    curve = fr_key + "curve_" + prefix

    train = h.get(prefix, None)
    val = h.get(val_key + prefix, None)
    fr_target = h.get(target, None)

    phase_re = re.compile(phase)
    fr_phase = list(chain.from_iterable(
        h[k] for k in h.keys()
        if phase_re.search(k) and isinstance(h[k], list)
    ))

    curve_re = re.compile(curve)
    curve_keys = [k for k in h.keys() if curve_re.search(k)]

    fr_curve = [(int(k.rsplit("_", 1)[-1]), h[k]) for k in curve_keys]

    train_path: Path | None = None
    val_path: Path | None = None
    fr_path: Path | None = None
    all_path: Path | None = None

    epoch_start = 0
    phase_intervals = []
    for ph in training_cfg.phases:
        start = epoch_start
        end = epoch_start + ph.epochs
        phase_intervals.append((start, end, ph.__class__.__name__))
        epoch_start = end

    # legenda per fasi
    legend_patches = [mpatches.Patch(color=phase_colors[ph_type], alpha=0.1, label=phase_labels[ph_type])
        for ph_type in {ph_type for _, _, ph_type in phase_intervals}]

    # --- 1) Train plot ---
    if train is not None and len(train) > 0:
        y = np.asarray(train, dtype=float)
        x = np.arange(1, len(y) + 1)

        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
        plt.plot(x, y)
        # linee verticali per inizio fase
        for _, end, _ in phase_intervals:
            plt.axvline(end, color='k', linestyle='--', linewidth=0.8)

        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_patches)

        plt.xlabel("Epoch")
        plt.ylabel(prefix)
        plt.title(f"Training {prefix}")
        plt.grid(True)

        train_path = run_dir / f"train_{prefix}.jpg"
        plt.savefig(train_path, dpi=300, bbox_inches="tight")
        plt.close()

    # --- 2) Val plot ---
    if val is not None and len(val) > 0:
        y = np.asarray(val, dtype=float)
        x = np.arange(1, len(y) + 1)

        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
        plt.plot(x, y)
        # linee verticali per inizio fase
        for _, end, _ in phase_intervals:
            plt.axvline(end, color='k', linestyle='--', linewidth=0.8)

        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_patches)

        plt.xlabel("Epoch")
        plt.ylabel(val_key + prefix)
        plt.title(f"Validation {val_key + prefix}")
        plt.grid(True)

        val_path = run_dir / f"{val_key + prefix}.jpg"
        plt.savefig(val_path, dpi=300, bbox_inches="tight")
        plt.close()

    # --- 3) Free-running plot ---
    if fr_phase is not None and any(not np.isnan(v) for v in fr_phase):

        x = [epoch + 1 for epoch, v in enumerate(fr_phase) if not np.isnan(v)]
        y = [v for v in fr_phase if not np.isnan(v)]

        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
        plt.plot(x, y)
        # linee verticali per inizio fase
        for _, end, _ in phase_intervals:
            plt.axvline(end, color='k', linestyle='--', linewidth=0.8)

        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_patches)

        plt.xlabel("Epoch")
        plt.ylabel(phase)
        plt.title(f"Free-running {phase}")
        plt.grid(True)

        fr_path = run_dir / f"{phase}.jpg"
        plt.savefig(fr_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    if fr_target is not None and any(not np.isnan(v) for v in fr_target):

        x = [epoch + 1 for epoch, v in enumerate(fr_target) if not np.isnan(v)]
        y = [v for v in fr_target if not np.isnan(v)]

        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
        plt.plot(x, y)
        # linee verticali per inizio fase
        for _, end, _ in phase_intervals:
            plt.axvline(end, color='k', linestyle='--', linewidth=0.8)

        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_patches)

        plt.xlabel("Epoch")
        plt.ylabel(target)
        plt.title(f"Free-running {target}")
        plt.grid(True)

        fr_path = run_dir / f"{target}.jpg"
        plt.savefig(fr_path, dpi=300, bbox_inches="tight")
        plt.close()
    

    if fr_curve is not None:
        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)

        for k,c in fr_curve:
            if c is not None and any(not np.isnan(v) for v in c):
                x = [epoch + 1 for epoch, v in enumerate(c) if not np.isnan(v)]
                y = [v for v in c if not np.isnan(v)]
                plt.plot(x, y, label=f"{k} horizons")

        # linee verticali per inizio fase
        for _, end, _ in phase_intervals:
            plt.axvline(end, color='k', linestyle='--', linewidth=0.8)

        plt.legend(
            handles=plt.gca().get_legend_handles_labels()[0] + legend_patches,
            fontsize=7
        )

        plt.xlabel("Epoch")
        plt.ylabel(curve)
        plt.title(f"Free-running {curve}")
        plt.grid(True)

        fr_path = run_dir / f"{curve}.jpg"
        plt.savefig(fr_path, dpi=300, bbox_inches="tight")
        plt.close()
                

    # --- 4) Combined plot  ---
    has_any = (
        (train is not None and len(train) > 0) or
        (val is not None and len(val) > 0) or
        (fr_phase is not None and any(not np.isnan(v) for v in fr_phase))
    )
    if has_any:
        plt.figure(figsize=(8,5))

        # sfondi per fase
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)

        # plot train/val/fr
        if train is not None and len(train) > 0:
            y = np.asarray(train, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=f"train_{prefix}")

        if val is not None and len(val) > 0:
            y = np.asarray(val, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=f"{val_key + prefix}")

        if fr_phase is not None and any(not np.isnan(v) for v in fr_phase):
            x = [epoch + 1 for epoch, v in enumerate(fr_phase) if not np.isnan(v)]
            y = [v for v in fr_phase if not np.isnan(v)]
            plt.plot(x, y, label=f"{phase}")

        # linee verticali per inizio fase
        for _, end, _ in phase_intervals:
            plt.axvline(end, color='k', linestyle='--', linewidth=0.8)

        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_patches, fontsize=7)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss curves (TF + Free-running)")
        plt.grid(True)

        all_path = run_dir / f"{prefix}_all.jpg"
        plt.savefig(all_path, dpi=300, bbox_inches="tight")
        plt.close()

    return train_path, val_path, fr_path, all_path
