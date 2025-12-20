from pathlib import Path
from datetime import datetime
import numpy as np
import json
from .plot import save_loss_plots_keras

def make_run_output_dir(model_cfg, root_dir: str | Path) -> Path:
    root_dir = Path(root_dir)
    model_type = str(model_cfg.type).upper()
    variant = str(model_cfg.variant)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_cfg.name}__{ts}"

    run_dir = root_dir / model_type / variant / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_outputs(
    splits,
    pred,
    model_cfg,
    history,
    eval_cfg: dict | None = None,
) -> Path:
    eval_cfg = eval_cfg or {}
    sample_index = int(eval_cfg.get("sample_index", 0))
    root_dir = eval_cfg.get("predictions_dir", "predictions")

    run_dir = make_run_output_dir(model_cfg=model_cfg, root_dir=root_dir)

    save_loss_plots_keras(run_dir,history)

    np.savez_compressed(
        run_dir / "data_splits.npz",
        X_test=splits.X_test,
        Y_test=splits.Y_test,
    )

    np.savez_compressed(
        run_dir / "predictions.npz",
        pred=np.asarray(pred),
        model_type=str(model_cfg.type),
        variant=str(model_cfg.variant),
        name=str(model_cfg.name),
    )

    meta = {
        "experiment_name": model_cfg.name,
        "model_type": str(model_cfg.type),
        "variant": str(model_cfg.variant),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_index": sample_index,
        "x_test_shape": list(splits.X_test.shape),
        "y_test_shape": list(splits.Y_test.shape),
        "pred_shape": list(np.asarray(pred).shape),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved artifacts to: {run_dir}")
    return run_dir
