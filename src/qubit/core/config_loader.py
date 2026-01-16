from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast, List
import yaml

from ..model.model_config import ModelConfig
from ..model.rnn_config import RNNConfig
from ..model.transformer_config import TransformerConfig
from ..model.compile_config import CompileConfig
from ..model.inference_config import InferenceConfig
from ..model.training_config import TrainingConfig
from ..model.fr_eval_config import *
from ..model.plot_config import PlotConfig
from ..model.phase_config import *


from .error import *

from ..enums.phase_name import PhaseName
from ..enums.split_name import SplitName
from ..enums.model_type import ModelType
from ..enums.model_variant import ModelVariant
from ..enums.decoder_mode import DecoderMode

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]  # core -> qubit -> src -> qubit-evolution-dl


def load_yaml(path: Path | str) -> Dict[str, Any]:
    path = Path("configs/" + str(path) + ".yaml").expanduser().resolve()
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
    save_model = m.get("save_model", False)

    try:
        model_type = ModelType(m["type"])
        variant = ModelVariant(m["variant"])
        decoder_mode = DecoderMode(m["decoder_mode"])
    except KeyError as e:
        raise ValueError(f"Missing required field: model.{e.args[0]}") from None
    except ValueError as e:
        raise ValueError(f"Invalid model config: {e}") from None

    # Parse the  config dict into a dataclass (dict is unpacked as kwargs).
    # common blocks
    params_dict = m.get("params", {}) or {}
    compile_cfg = CompileConfig(**(m.get("compile", {}) or {}))
    inference_cfg = InferenceConfig(**(m.get("inference", {}) or {}))

    if model_type == ModelType.LSTM:
        if variant in ModelVariant:
            params = RNNConfig(**params_dict)
        else:
            raise ValueError(f"Unknown RNN variant: {variant}")

    elif model_type == ModelType.TRN:
        if variant in ModelVariant:
            params = TransformerConfig(**params_dict)
        else:
            raise ValueError(f"Unknown Transformer variant: {variant}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return ModelConfig(
        name=name,
        save_model = save_model,
        type=model_type,
        variant=variant,
        decoder_mode=decoder_mode,
        compile=compile_cfg,
        inference=inference_cfg,
        params=params,
    )

def load_run_config(path: str) -> dict:

    # dictionary from yaml file 
    cfg = load_yaml(path)

    # verify if in the dicitionary are the mandatory key
    for k in ["data", "model", "training","plot"]:
        if k not in cfg:
            raise ValueError(f"Missing '{k}' in run config")
    return cfg

def parse_phase(d: Dict[str, Any]) -> PhaseConfig:
    if "name" not in d:
        raise ValueError("Phase config missing 'name' field")

    name = PhaseName(d["name"])  # valida automaticamente (ValueError se typo)

    if name == PhaseName.TEACHER_FORCING:
        return TeacherForcingPhase(**d)
    if name == PhaseName.MASKED_MODELING:
        return MaskedModelingPhase(**d)
    if name == PhaseName.SCHEDULED_SAMPLING:
        return ScheduledSamplingPhase(**d)
    if name == PhaseName.FULL_AUTOREGRESSIVE:
        return FullAutoregressivePhase(**d)

    raise ValueError(f"Unsupported phase name: {name}")


def validate_masked_modeling_phase(p: MaskedModelingPhase) -> None:
    if p.mask_mode == MaskMode.CONSTANT:
        if p.mask_value is None:
            raise ValueError("MaskedModelingPhase: mask_mode=CONSTANT need mask_value.")

    elif p.mask_mode == MaskMode.NOISE:
        if p.noise_sigma is None:
            raise ValueError("MaskedModelingPhase: mask_mode=NOISE need noise_sigma.")
        
    elif p.mask_mode == MaskMode.ZERO:
        pass

    else:
        raise ValueError(f"MaskedModelingPhase: mask_mode non supportato: {p.mask_mode}")


def load_training_config(t: Dict[str, Any]) -> TrainingConfig:
    verbose = t.get("verbose", 1)
    curriculum = t.get("curriculum", [])
    batch_size = t.get("batch_size", 32)
    phases_raw = t.get("phases", []) or []
    phases = [parse_phase(p) for p in phases_raw]
    epochs = sum(p.epochs for p in phases)

    for p in phases:
        if isinstance(p, MaskedModelingPhase):
            validate_masked_modeling_phase(p)    

    fr_eval_raw = t.get("fr_eval", {}) or {}

    # split enum-safe
    split = SplitName(fr_eval_raw.get("split", "val")) 
    enabled = bool(fr_eval_raw.get("enabled", False))

    # parse probes 
    probes_raw = fr_eval_raw.get("probes")
    if probes_raw is  None:
        raise ValueError("fr_eval config must define probes field.")
    
    probes = tuple(_parse_fr_eval_probe(cast(Dict[str, Any], p)) for p in probes_raw)


    if enabled and len(probes) == 0:
        raise ValueError("if fr_eval.enabled=true requires at least one probe")

    fr_eval = FrEvalConfig(
        enabled=enabled,
        split=split,
        probes=cast(tuple[FrEvalProbeConfig],probes),
    )

    return TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        phases=phases,
        fr_eval=fr_eval,
        verbose=verbose,
        curriculum=curriculum,
    )


def _parse_fr_eval_probe(p: Dict[str, Any]) -> FrEvalProbeConfig:
    out_steps = p.get("out_steps")

    if out_steps is None:
        raise ValueError(f"FrEvalProbeConfig '{p.get('name')}' must define 'out_steps' ")

    return FrEvalProbeConfig(
        name=cast(str, p["name"]),
        every_epochs=cast(EveryEpochs, p.get("every_epochs")),
        out_steps=cast(OutStepsSpec, out_steps),
        p_eval=cast(float, p.get("p_eval")),
    )

def _parse_sample_index(v: Any) -> List[int]:
    if v is None:
        return [0]

    if isinstance(v, int):
        return [v]

    if isinstance(v, (list, tuple)):
        out: List[int] = []
        for x in v:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str) and x.strip():
                out.append(int(x.strip()))
            else:
                raise TypeError(f"sample_index contains unsupported type: {type(x)}")
        return out

    raise TypeError(f"Unsupported sample_index type: {type(v)}")


def load_plot_config(p: Dict[str, Any]) -> PlotConfig:

    pred_all = bool(p.get("pred_all", True))
    save_plots = bool(p.get("save_plots", True))
    save_artifacts = bool(p.get("save_artifacts", True))
    sample_index = _parse_sample_index(p.get("sample_index", [0]))



    return PlotConfig(
        pred_all=pred_all,
        sample_index=sample_index,
        save_plots=save_plots,
        save_artifacts=save_artifacts,
    )