'''
from pathlib import Path
from datetime import datetime
import numpy as np


def make_run_output_dir(
    model_cfg,
    root_dir: str | Path = "artifacts",
) -> Path:
    root_dir = Path(root_dir)
    model_type = str(model_cfg.type).upper()
    variant = str(model_cfg.variant)

    # timestamp (locale)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # nome cartella run
    run_name = f"{model_cfg.name}__{ts}"

    run_dir = root_dir / model_type / variant / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_outputs(
    splits,
    pred,
    model_cfg,
    eval_cfg: dict | None = None,
):
    eval_cfg = eval_cfg or {}
    sample_index = int(eval_cfg.get("sample_index", 0))

    run_dir = make_run_output_dir(
        model_cfg=model_cfg,
        root_dir=eval_cfg.get("predictions_dir", "predictions"),
    )

    # 1) salva predizione (utile per debug/riproducibilità)
    np.save(run_dir / "prediction.npy", pred)

    # 2) salva anche un esempio di input/target se vuoi
    # np.save(run_dir / "x_sample.npy", splits.X_train[sample_index:sample_index+1])
    # np.save(run_dir / "y_sample.npy", splits.Y_train[sample_index:sample_index+1])

    # 3) salva plot in quella cartella
    # ==> ti consiglio di modificare generate_all_plots per accettare output_dir
    generate_all_plots(
        splits,
        transformer_prediction=pred if model_cfg.type == "TRN" else None,
        rnn_prediction=pred if model_cfg.type == "RNN" else None,
        sample_index=sample_index,
        output_dir=str(run_dir),   # <--- aggiungi questo parametro nella funzione
    )

    print(f"✅ Saved outputs to: {run_dir}")
    return run_dir

''' 