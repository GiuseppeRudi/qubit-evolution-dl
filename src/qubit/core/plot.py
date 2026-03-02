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

HORIZON_TEXT_COLOR = "purple"

phase_labels = {
    "TeacherForcingPhase": "Teacher Forcing",
    "MaskedModelingPhase": "Masked Modeling",
    "ScheduledSamplingPhase": "Scheduled Sampling",
    "FullAutoregressivePhase": "Full Autoregressive"
}

def save_loss_plots_keras(
    run_dir: Path,
    history,
    training_cfg: TrainingConfig,
    output_seq_len: int,
    val_key: str = "val_",
    prefix: str = "loss",
):

    run_dir = Path(str(run_dir) + "/loss_plots")
    run_dir.mkdir(parents=True, exist_ok=True)

    h = history.history
    print(h)
    
    fr_key = training_cfg.fr_eval.split + "_fr_"

    target = fr_key + "target_" + prefix + "_" + str(output_seq_len)
    phase = fr_key + "phase_" + prefix
    curve = fr_key + "curve_" + prefix

    train = h.get(prefix, None)
    val = h.get(val_key + prefix, None)
    fr_target = h.get(target, None)

    # fr_phase
    phase_re = re.compile(phase)
    def horizon_of(k):
        m = re.search(r'_(\d+)$', k)
        return int(m.group(1)) if m else -1  # o 0/None come preferisci

    phase_keys = sorted(
        [k for k in h.keys() if phase_re.search(k) and isinstance(h[k], list)],
        key=horizon_of
    )
    
    fr_phase_data = []
    phase_horizons = {}
    
    epoch_offset = 0
    for k in phase_keys:
        horizon_match = re.search(r'_(\d+)$', k)
        horizon = int(horizon_match.group(1)) if horizon_match else None
        values = h[k]
        
        for i, v in enumerate(values):
            if not np.isnan(v):
                global_epoch = epoch_offset + i
                fr_phase_data.append((global_epoch + 1, v, horizon))
                phase_horizons[global_epoch + 1] = horizon
        
        epoch_offset += len(values)
    
    max_epoch = max((e for e, _, _ in fr_phase_data), default=0)
    fr_phase = [np.nan] * max_epoch
    for epoch_1based, v, _ in fr_phase_data:
        fr_phase[epoch_1based - 1] = v

    # fr_curve
    curve_re = re.compile(curve)
    curve_keys = [k for k in h.keys() if curve_re.search(k)]

    fr_curve = [(int(k.rsplit("_", 1)[-1]), h[k]) for k in curve_keys]

    epoch_start = 0
    phase_intervals = []
    for ph in training_cfg.phases:
        start = epoch_start
        end = epoch_start + ph.epochs
        phase_intervals.append((start, end, ph.__class__.__name__))
        epoch_start = end

    # legend for each phases
    legend_patches = [mpatches.Patch(color=phase_colors[ph_type], alpha=0.1, label=phase_labels[ph_type])
        for ph_type in {ph_type for _, _, ph_type in phase_intervals}]
    
    # legend for each phases + Horizons indicator
    combined_legend_patches = legend_patches + [mpatches.Patch(color=HORIZON_TEXT_COLOR, alpha=0.3, label="Phase horizons")]

    # Train plot
    if train is not None and len(train) > 0:
        y = np.asarray(train, dtype=float)
        x = np.arange(1, len(y) + 1)

        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
        plt.plot(x, y)
        # vertical lines for start of phase
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

    # Val plot
    if val is not None and len(val) > 0:
        y = np.asarray(val, dtype=float)
        x = np.arange(1, len(y) + 1)

        plt.figure()
        # background for phase
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
        plt.plot(x, y)
        
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

    # Free-running-phase plot
    if fr_phase is not None and any(not np.isnan(v) for v in fr_phase):

        x = [epoch + 1 for epoch, v in enumerate(fr_phase) if not np.isnan(v)]
        y = [v for v in fr_phase if not np.isnan(v)]

        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
        plt.plot(x, y)

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
    
    # Free-running-target plot
    if fr_target is not None and any(not np.isnan(v) for v in fr_target):

        x = [epoch + 1 for epoch, v in enumerate(fr_target) if not np.isnan(v)]
        y = [v for v in fr_target if not np.isnan(v)]

        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
        plt.plot(x, y)

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
    
    # Free-running-curve plot
    if fr_curve is not None and len(fr_curve)!=0:
        plt.figure()
        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)

        for k,c in fr_curve:
            if c is not None and any(not np.isnan(v) for v in c):
                x = [epoch + 1 for epoch, v in enumerate(c) if not np.isnan(v)]
                y = [v for v in c if not np.isnan(v)]
                plt.plot(x, y, label=f"{k} horizons")

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
                

    # train/val/fr plot combined
    has_any = (
        (train is not None and len(train) > 0) or
        (val is not None and len(val) > 0) or
        (fr_phase is not None and any(not np.isnan(v) for v in fr_phase))
    )
    if has_any:
        plt.figure(figsize=(8,5))

        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
            
            horizons = [phase_horizons.get(e) for e in range(start, end) if e in phase_horizons]
            if horizons:
                h_val = max(set(horizons), key=horizons.count)  # moda
                mid = (start + end) / 2
                plt.text(mid, 0.02, str(h_val), transform=plt.gca().get_xaxis_transform(),
                        ha='center', fontsize=9, color=HORIZON_TEXT_COLOR, fontweight='bold')

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

        for _, end, _ in phase_intervals:
            plt.axvline(end, color='k', linestyle='--', linewidth=0.8)

        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + combined_legend_patches, fontsize=7)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss curves (TF + Free-running)")
        plt.grid(True)

        all_path = run_dir / f"{prefix}_all.jpg"
        plt.savefig(all_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    # curve/target plot combined
    has_curve_target = (
        (fr_target is not None and any(not np.isnan(v) for v in fr_target)) or
        (fr_curve is not None and len(fr_curve) != 0)
    )
    
    if has_curve_target:
        plt.figure(figsize=(8, 5))

        for start, end, ph_type in phase_intervals:
            plt.axvspan(start, end, color=phase_colors[ph_type], alpha=0.1)
            
            horizons = [phase_horizons.get(e) for e in range(start, end) if e in phase_horizons]
            if horizons:
                h_val = max(set(horizons), key=horizons.count)
                mid = (start + end) / 2
                plt.text(mid, 0.02, str(h_val), transform=plt.gca().get_xaxis_transform(),
                        ha='center', fontsize=9, color=HORIZON_TEXT_COLOR, fontweight='bold')

        if fr_curve is not None and len(fr_curve) != 0:
            for k, c in fr_curve:
                if c is not None and any(not np.isnan(v) for v in c):
                    x = [epoch + 1 for epoch, v in enumerate(c) if not np.isnan(v)]
                    y = [v for v in c if not np.isnan(v)]
                    plt.plot(x, y, label=f"{k} horizons")

        if fr_target is not None and any(not np.isnan(v) for v in fr_target):
            x = [epoch + 1 for epoch, v in enumerate(fr_target) if not np.isnan(v)]
            y = [v for v in fr_target if not np.isnan(v)]

            target_horizon_match = re.search(r'_(\d+)$', target)
            target_horizon = target_horizon_match.group(1) if target_horizon_match else str(output_seq_len)
            plt.plot(x, y, label=f"{target_horizon} horizons", linewidth=2, linestyle='--')

        for _, end, _ in phase_intervals:
            plt.axvline(end, color='k', linestyle='--', linewidth=0.8)

        handles, _ = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles + combined_legend_patches, fontsize=7)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Free-running {curve} + {target}")
        plt.grid(True)

        curve_target_path = run_dir / f"{curve}_{target}_combined.jpg"
        plt.savefig(curve_target_path, dpi=300, bbox_inches="tight")
        plt.close()
