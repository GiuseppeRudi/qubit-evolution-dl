from dataclasses import dataclass
from typing import Literal, Union

from ..enums.phase_name import PhaseName

from ..strategies.base_strategy import TrainingStrategy

@dataclass(frozen=True)
class TeacherForcingPhase:
    name: Literal[PhaseName.TEACHER_FORCING] 
    epochs: int 
    learning_rate : float = -1
    clip_norm : float = -1 


@dataclass(frozen=True)
class MaskedModelingPhase:
    name: Literal[PhaseName.MASKED_MODELING] 
    epochs: int 
    mask_prob: float 
    learning_rate : float = -1
    clip_norm : float = -1 

@dataclass(frozen=True)
class ScheduledSamplingPhase:
    name: Literal[PhaseName.SCHEDULED_SAMPLING] 
    epochs: int 
    tf_ratio_start: float 
    tf_ratio_end: float 
    learning_rate : float = -1
    clip_norm : float = -1 

@dataclass(frozen=True)
class FullAutoregressivePhase:
    name: Literal[PhaseName.FULL_AUTOREGRESSIVE] 
    epochs: int 
    gradient_through_time: bool
    learning_rate : float = -1
    clip_norm : float = -1 


PhaseConfig = Union[TeacherForcingPhase, MaskedModelingPhase, ScheduledSamplingPhase, FullAutoregressivePhase]


@dataclass
class Phase:
    cfg: PhaseConfig
    strategy: TrainingStrategy
