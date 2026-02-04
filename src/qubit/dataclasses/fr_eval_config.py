from dataclasses import dataclass
from typing import Union, Literal

from ..enums.split_name import SplitName

from ..utils.config_values import END_OF_PHASE, OUT_STEPS_SPEC

EveryEpochs  = Union[int, END_OF_PHASE]
OutStepsSpec = Union[tuple[float], OUT_STEPS_SPEC]

@dataclass(frozen=True)
class FrEvalProbeConfig:
    name: str
    every_epochs: EveryEpochs 

    # can be a list of horizon values or a string that derminate from where to take the outpsteps
    out_steps: OutStepsSpec
    p_eval: float


@dataclass(frozen=True)
class FrEvalConfig:
    enabled: bool
    split: SplitName
    batch_size: int
    # used tuple because this type of object is immutable instead list can be mutable
    probes: tuple[FrEvalProbeConfig, ...]
