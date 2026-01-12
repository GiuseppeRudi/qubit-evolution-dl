from pathlib import Path
from datetime import datetime
from typing import List
import numpy as np
import json
from .plot import save_loss_plots_keras

from ..enums.model_type import ModelType
from ..enums.model_variant import ModelVariant
from ..enums.decoder_mode import DecoderMode

from ..model.model_config import ModelConfig
from ..model.plot_config import PlotConfig
from ..model.training_config import TrainingConfig
from ..model.phase_config import PhaseConfig

import sys
from ..core.utils import Tee

def save_log(run_dir : Path):

    log_path = run_dir / "train.log"
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

def make_run_output_dir(model_cfg : ModelConfig) -> Path:
    root_dir = Path("predictions")
    model_type : ModelType = model_cfg.type
    variant : ModelVariant = model_cfg.variant
    decoder_mode : DecoderMode = model_cfg.decoder_mode

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_cfg.name}__{ts}"

    run_dir = root_dir / model_type / variant / decoder_mode / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_outputs(
    splits,
    pred,
    model_cfg : ModelConfig,
    feat_names,
    history,
    run_dir,
    fr_key: str,
    plot_cfg: PlotConfig,
    phases : List[PhaseConfig],
    model = None,
) -> Path:
    sample_index = plot_cfg.sample_index
    save_model = model_cfg.save_model
    save_plots = plot_cfg.save_plots
    save_artifacts  = plot_cfg.save_artifacts

    

    # if save_model and model is not None:
        # TODO with custom model we dont use directly model.save
        # for this reasons we save the weights and later modify the code to recreate the model from the weights 
        # model.save_weights(run_dir / "weights.h5")  # o .weights.h5
        # model.save(run_dir/"model.keras")

    if history is not None and save_plots:
        save_loss_plots_keras(run_dir,history,phases,fr_key)

    if save_artifacts:
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
            "feature_names": feat_names,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    
        print(f"Saved artifacts to: {run_dir}")
        
    return run_dir
