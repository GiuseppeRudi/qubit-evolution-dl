from dataclasses import dataclass
from typing import Literal, Union

from ..enums.phase_name import PhaseName

@dataclass(frozen=True)
class TeacherForcingPhase:
    name: Literal[PhaseName.TEACHER_FORCING] 
    epochs: int 
    tf_ratio: float 


@dataclass(frozen=True)
class MaskedModelingPhase:
    name: Literal[PhaseName.MASKED_MODELING] 
    epochs: int 
    mask_prob: float 


@dataclass(frozen=True)
class ScheduledSamplingPhase:
    name: Literal[PhaseName.SCHEDULED_SAMPLING] 
    epochs: int 
    tf_ratio_start: float 
    tf_ratio_end: float 


PhaseConfig = Union[TeacherForcingPhase, MaskedModelingPhase, ScheduledSamplingPhase]
