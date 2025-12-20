from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_loss_plots_keras(
    run_dir: str | Path,
    history,
    train_key: str = "loss",
    val_key: str = "val_loss",
    prefix: str = "loss",
) -> tuple[Path | None, Path | None]:

    run_dir = Path(run_dir)

    if not hasattr(history, "history") or not isinstance(history.history, dict):
        raise TypeError("Expected a Keras History object (returned by model.fit), with a .history dict")

    h = history.history

    train = h.get(train_key, None)
    val = h.get(val_key, None)

    train_path: Path | None = None
    val_path: Path | None = None

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

    return train_path, val_path
