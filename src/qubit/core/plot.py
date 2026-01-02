from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_loss_plots_keras(
    run_dir: str | Path,
    history,
    train_key: str = "loss",
    val_key: str = "val_loss",
    fr_key: str = "test_fr_loss",   
    prefix: str = "loss",
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """
    Save 4 plot:
      - {prefix}_train.png  : train_key (es. loss)
      - {prefix}_val.png    : val_key (es. val_loss)
      - {prefix}_fr.png     : fr_key (es. test_fr_loss)
      - {prefix}_all.png    : combined (train + val + fr)

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

        train_path = run_dir / f"{prefix}_train.png"
        plt.savefig(train_path, dpi=150, bbox_inches="tight")
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

        val_path = run_dir / f"{prefix}_val.png"
        plt.savefig(val_path, dpi=150, bbox_inches="tight")
        plt.close()

    # --- 3) Free-running plot ---
    if fr is not None and len(fr) > 0:
        y = np.asarray(fr, dtype=float)
        x = np.arange(1, len(y) + 1)

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Epoch")
        plt.ylabel(fr_key)
        plt.title(f"Free-running {fr_key}")
        plt.grid(True)

        fr_path = run_dir / f"{prefix}_fr.png"
        plt.savefig(fr_path, dpi=150, bbox_inches="tight")
        plt.close()

    # --- 4) Combined plot  ---
    has_any = (
        (train is not None and len(train) > 0) or
        (val is not None and len(val) > 0) or
        (fr is not None and len(fr) > 0)
    )
    if has_any:
        plt.figure()

        if train is not None and len(train) > 0:
            y = np.asarray(train, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=f"train ({train_key})")

        if val is not None and len(val) > 0:
            y = np.asarray(val, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=f"val ({val_key})")

        if fr is not None and len(fr) > 0:
            y = np.asarray(fr, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=f"free-running ({fr_key})")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss curves (TF + Free-running)")
        plt.grid(True)
        plt.legend()

        all_path = run_dir / f"{prefix}_all.png"
        plt.savefig(all_path, dpi=150, bbox_inches="tight")
        plt.close()

    return train_path, val_path, fr_path, all_path
