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
    path = Path("configs/" + path + ".yaml").expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping (dict) in {path}, got {type(data).__name__}")

    # opzionale: se vuoi anche assicurarti che le chiavi siano str
    if not all(isinstance(k, str) for k in data.keys()):
        raise TypeError(f"Expected dict with str keys in {path}")

    return cast(Dict[str, Any], data)


def load_model_config(m: Dict[str, Any]) -> ModelConfig:

    # free label (for distinction)
    name = m.get("name", "experiment")

    # discriminants (fixed)
    model_type = m.get("type")
    variant = m.get("variant")

    if model_type is None:
        raise ValueError("Missing required field: model.type")
    if variant is None:
        raise ValueError("Missing required field: model.variant")


    # Parse the  config dict into a dataclass (dict is unpacked as kwargs).
    # common blocks
    compile_cfg = CompileConfig(**(m.get("compile", {}) or {}))
    training_cfg = TrainingConfig(**(m.get("training", {}) or {}))
    params_dict = m.get("params", {}) or {}

    model_type_norm = str(model_type).strip().upper()
    variant_norm = str(variant).strip().upper()

    #TODO create a enum without this harcoding value, the same thing in ModelConfig
    if model_type_norm == "RNN":
        if variant_norm in {"SEQ2SEQ", "ALTRE..", "ALTRE.."}:
            params = RNNConfig(**params_dict)
        else:
            raise ValueError(f"Unknown RNN variant: {variant}")

    elif model_type_norm in {"TRN"}:
        if variant_norm in {"ENCODERDENSE", "ALTRE.."}:
            params = TransformerConfig(**params_dict)
        else:
            raise ValueError(f"Unknown Transformer variant: {variant}")
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

    # dictionary from yaml file 
    cfg = load_yaml(path)

    # verify if in the dicitionary are the mandatory key
    for k in ["data", "model", "training"]:
        if k not in cfg:
            raise ValueError(f"Missing '{k}' in run config")
    return cfg
