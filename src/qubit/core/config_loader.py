from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast
import yaml

from ..model.model_config import ModelConfig
from ..model.rnn_config import RNNConfig
from ..model.training_config import TrainingConfig
from ..model.transformer_config import TransformerConfig
from ..model.compile_config import CompileConfig



def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]  # core -> qubit -> src -> qubit-evolution-dl


def load_yaml(path: Path | str) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping (dict) in {path}, got {type(data).__name__}")

    # opzionale: se vuoi anche assicurarti che le chiavi siano str
    if not all(isinstance(k, str) for k in data.keys()):
        raise TypeError(f"Expected dict with str keys in {path}")

    return cast(Dict[str, Any], data)


# return a dictionary with dataset config
def load_dataset_config(cfg_path: Path | str) -> Dict[str, Any]:
    root = get_project_root()
    cfg = load_yaml(cfg_path)

    csv_path = cfg["dataset"]["csv_path"]

    # if the path is not start from project root, make it start from it
    cfg["dataset"]["csv_path"] = str((root / csv_path).resolve()) if not str(csv_path).startswith("/") else csv_path

    print(cfg)

    return cfg



def load_model_config(m: Dict[str, Any]) -> ModelConfig:
    """
    Build a ModelConfig from a dict representing the 'model' section.
    Expected keys: name (optional), type, variant, params, compile, training.
    """
    if not isinstance(m, dict):
        raise TypeError(f"Expected dict for model config, got {type(m)}")

    # free label (for logging/saving)
    name = m.get("name", "experiment")

    # discriminants (fixed)
    model_type = m.get("type")
    variant = m.get("variant")

    if model_type is None:
        raise ValueError("Missing required field: model.type")
    if variant is None:
        raise ValueError("Missing required field: model.variant")

    # common blocks
    compile_cfg = CompileConfig(**(m.get("compile", {}) or {}))
    training_cfg = TrainingConfig(**(m.get("training", {}) or {}))
    params_dict = m.get("params", {}) or {}

    # normalize strings
    model_type_norm = str(model_type).strip().upper()
    variant_norm = str(variant).strip().upper()

    # choose params class
    if model_type_norm == "RNN":
        # accetta "Seq2Seq" ecc.
        if variant_norm in {"SEQ2SEQ", "SEQ_2_SEQ", "SEQ-2-SEQ"}:
            params = RNNConfig(**params_dict)
        else:
            raise ValueError(f"Unknown RNN variant: {variant}")
        # forza type a valori canonici
        model_type_norm = "RNN"

    elif model_type_norm in {"TRN", "TRANSFORMER"}:
        # per il tuo transformer attuale è più corretto usare "ENCODERDENSE" o simili,
        # ma lasciamo compatibilità anche con vecchi nomi:
        if variant_norm in {"ENCODERDENSE", "ENCODER_DENSE", "ENCODER", "ENCODER_ONLY", "SEQ2SEQ"}:
            params = TransformerConfig(**params_dict)
        else:
            raise ValueError(f"Unknown Transformer variant: {variant}")
        model_type_norm = "TRN"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return ModelConfig(
        name=name,
        type=model_type_norm,
        variant=variant_norm,
        compile=compile_cfg,
        training=training_cfg,
        params=params,
    )


def load_run_config(path: str) -> dict:
    cfg = load_yaml(path)
    # valida campi minimi
    for k in ["data", "model", "training"]:
        if k not in cfg:
            raise ValueError(f"Missing '{k}' in run config")
    return cfg
