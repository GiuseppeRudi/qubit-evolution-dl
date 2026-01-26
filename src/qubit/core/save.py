from pathlib import Path
from datetime import datetime
import shutil
from xml.parsers.expat import model
import numpy as np
import json

from .plot import save_loss_plots_keras

from ..enums.model_type import ModelType
from ..enums.model_variant import ModelVariant
from ..enums.decoder_mode import DecoderMode

from ..dataclasses.model_config import ModelConfig
from ..dataclasses.plot_config import PlotConfig
from ..dataclasses.training_config import TrainingConfig

from ..utils.config_values import LOG_PATH, PREDICTION_PATH
from ..utils.utils import BufferedLogger, finish_log


def make_run_output_dir(model_cfg : ModelConfig) -> Path:

    root_dir = Path(PREDICTION_PATH)
    model_type : ModelType = model_cfg.type
    variant : ModelVariant = model_cfg.variant
    decoder_mode : DecoderMode = model_cfg.decoder_mode

    # take the datatime at the moment of creation
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # take the name of the model and append the timestamp
    run_name = f"{model_cfg.name}__{ts}"

    run_dir = root_dir / model_type / variant / decoder_mode / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_yaml(yaml_name: str, run_dir: Path):

    # path sorgente dello yaml
    src = Path("configs/" + yaml_name + ".yaml").expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"YAML not found: {src}")

    # path destinazione (stesso nome o nome custom)
    dst = run_dir / src.name

    # copia fisica del file
    shutil.copy2(src, dst)

def save_outputs(
    splits,
    pred, # shape(num_windows, output_seq_len, feature_dim)
    model_cfg : ModelConfig,
    feat_names,
    history,
    fr_key: str,
    plot_cfg: PlotConfig,
    training_cfg : TrainingConfig,
    output_seq_len: int,
    bufferedLogger: BufferedLogger, 
    yaml_name: str,
    model = None,
) -> Path:
    
    run_dir = make_run_output_dir(model_cfg)

    sample_index = plot_cfg.sample_index
    save_model = model_cfg.save_model
    save_plots = plot_cfg.save_plots
    save_artifacts  = plot_cfg.save_artifacts

    if save_model and model is not None:
        print("model built:", model.built)
        print("num weights:", len(model.weights))
        model.save_weights(run_dir / "model.weights.h5")

    if history is not None and save_plots:
        save_loss_plots_keras(run_dir,history,training_cfg,fr_key,output_seq_len)

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
            "model_type": model_cfg.type.value,
            "variant": model_cfg.variant.value,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sample_index": sample_index,
            "x_test_shape": list(splits.X_test.shape),
            "y_test_shape": list(splits.Y_test.shape),
            "pred_shape": list(np.asarray(pred).shape),
            "feature_names": feat_names,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"Saved artifacts to: {run_dir}")

    copy_yaml(yaml_name, run_dir)
    finish_log(bufferedLogger,run_dir)
        
    return run_dir
