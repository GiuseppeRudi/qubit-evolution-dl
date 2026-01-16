from dataclasses import dataclass
from typing import Optional, Sequence, Union, Literal

from ..enums.split_name import SplitName

EveryEpochs = Union[int, Literal["end_of_phase"]]
OutStepsSpec =Union[tuple[int], Literal["phase", "global"]]

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
    # used tuple because this type of object is immutable instead list can be mutable
    probes: tuple[FrEvalProbeConfig, ...]
