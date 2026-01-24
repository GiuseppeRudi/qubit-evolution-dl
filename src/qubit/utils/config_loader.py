from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast, List
import yaml

from ..dataclasses.model_config import ModelConfig
from ..dataclasses.rnn_config import RNNConfig
from ..dataclasses.transformer_config import TransformerConfig
from ..dataclasses.compile_config import CompileConfig
from ..dataclasses.inference_config import InferenceConfig
from ..dataclasses.training_config import TrainingConfig
from ..dataclasses.fr_eval_config import *
from ..dataclasses.plot_config import PlotConfig
from ..dataclasses.data_config import DataConfig
from ..dataclasses.phase_config import *

from ..utils.config_keys import *

from .error import *

from ..enums.phase_name import PhaseName
from ..enums.split_name import SplitName
from ..enums.model_type import ModelType
from ..enums.model_variant import ModelVariant
from ..enums.decoder_mode import DecoderMode

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]  # core -> qubit -> src -> qubit-evolution-dl


def load_yaml(yaml_name: str) -> Dict[str, Any]:
    
    # takes the absolute path 
    path = Path("configs/" + yaml_name + ".yaml").expanduser().resolve()
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # check if the data is a dictionary 
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping (dict) in {path}, got {type(data).__name__}")

    return cast(Dict[str, Any], data)


def load_model_config(m: Dict[str, Any]) -> ModelConfig:

    name = m[MODEL_NAME]
    save_model = m[SAVE_MODEL]
    model_type = ModelType(m[TYPE])
    variant = ModelVariant(m[VARIANT])
    decoder_mode = DecoderMode(m[DECODER_MODE])

    # parse the config dict into a dataclass (dict is unpacked as kwargs).
    params_dict = m[PARAMS]
    compile_cfg = CompileConfig(**m[COMPILE])
    inference_cfg = InferenceConfig(**m[INFERENCE])

    if model_type == ModelType.LSTM:
        if variant in ModelVariant: params = RNNConfig(**params_dict)
        else: raise ValueError(f"Unknown LSTM variant: {variant}")

    elif model_type == ModelType.TRN:
        if variant in ModelVariant: params = TransformerConfig(**params_dict)
        else: raise ValueError(f"Unknown TRN variant: {variant}")
    
    else: raise ValueError(f"Unknown model type: {model_type}")

    return ModelConfig(
        name = name,
        save_model = save_model,
        type = model_type,
        variant = variant,
        decoder_mode = decoder_mode,
        compile = compile_cfg,
        inference = inference_cfg,
        params = params,
    )

def load_run_config(yaml_name: str) -> dict:

    # dictionary from yaml file 
    cfg = load_yaml(yaml_name)

    # verify if in the dicitionary are the mandatory key
    for k in [DATA, MODEL, TRAINING, PLOT]:
        if k not in cfg:
            raise ValueError(f"Missing '{k}' in run config")
        
    return cfg

def parse_phase(d: Dict[str, Any]) -> PhaseConfig:

    if PHASE_NAME not in d:
        raise ValueError(f"Phase config missing '{PHASE_NAME}' field")

    name = PhaseName(d[PHASE_NAME]) 

    if name == PhaseName.TEACHER_FORCING:
        return TeacherForcingPhase(**d)
    if name == PhaseName.MASKED_MODELING:
        return MaskedModelingPhase(**d)
    if name == PhaseName.SCHEDULED_SAMPLING:
        return ScheduledSamplingPhase(**d)
    if name == PhaseName.FULL_AUTOREGRESSIVE:
        return FullAutoregressivePhase(**d)

    raise ValueError(f"Unsupported phase name: {name}")



def load_training_config(t: Dict[str, Any]) -> TrainingConfig:
    
    verbose = t[TRAINING_VERBOSE]
    curriculum = t[CURRICULUM]
    batch_size = t[BATCH_SIZE]
    prediction_mode = t[PREDICTION_MODE]

    # parse phases
    phases_raw = t[PHASES]
    phases = [parse_phase(p) for p in phases_raw]

    for p in phases:
        if isinstance(p, MaskedModelingPhase):
            validate_masked_modeling_phase(p)    

    epochs = sum(p.epochs for p in phases)

    # parse probes 
    fr_eval_raw = t[FR_EVAL]
    split = SplitName(fr_eval_raw[FR_EVAL_SPLIT]) 
    enabled = bool(fr_eval_raw[ENABLED])
    batch_size = int(fr_eval_raw[FR_BATCH_SIZE])

    probes_raw = fr_eval_raw[PROBES]
    if probes_raw is None:
        raise ValueError("fr_eval config must define probes field.")
    
    probes = tuple(_parse_fr_eval_probe(cast(Dict[str, Any], p)) for p in probes_raw)

    if enabled and len(probes) == 0:
        raise ValueError("if fr_eval.enabled=true requires at least one probe")

    fr_eval = FrEvalConfig(
        enabled=enabled,
        split=split,
        batch_size=batch_size,
        probes=cast(tuple[FrEvalProbeConfig],probes),
    )

    return TrainingConfig(
        prediction_mode=prediction_mode,
        batch_size=batch_size,
        epochs=epochs,
        phases=phases,
        fr_eval=fr_eval,
        verbose=verbose,
        curriculum=curriculum,
    )

def load_data_config(d: Dict[str, Any]) -> DataConfig:
    return DataConfig.from_dict(d)

def _parse_fr_eval_probe(p: Dict[str, Any]) -> FrEvalProbeConfig:
    out_steps = p[OUT_STEPS]

    if out_steps is None:
        raise ValueError(f"FrEvalProbeConfig '{p[PROBE_NAME]}' must define 'out_steps' ")

    return FrEvalProbeConfig(
        name=cast(str, p[PROBE_NAME]),
        every_epochs=cast(EveryEpochs, p[EVERY_EPOCHS]),
        out_steps=cast(OutStepsSpec, out_steps),
        p_eval=cast(float, p[P_EVAL]),
    )

def _parse_sample_index(v: Any) -> List[int]:
    if v is None: return [0]

    if isinstance(v, int): return [v]

    if isinstance(v, (list, tuple)):
        out: List[int] = []
        for x in v:
            if isinstance(x, int): out.append(x)
            elif isinstance(x, str) and x.strip(): out.append(int(x.strip()))
            else: raise TypeError(f"sample_index contains unsupported type: {type(x)}")
        return out

    raise TypeError(f"Unsupported sample_index type: {type(v)}")


def load_plot_config(p: Dict[str, Any]) -> PlotConfig:

    save_plots = bool(p[SAVE_PLOTS])
    save_artifacts = bool(p[SAVE_ARTIFACTS])
    sample_index = _parse_sample_index(p[SAMPLE_INDEX])

    return PlotConfig(
        sample_index=sample_index,
        save_plots=save_plots,
        save_artifacts=save_artifacts,
    )