from dataclasses import dataclass
from typing import Literal, Union

from ..enums.phase_name import PhaseName

@dataclass(frozen=True)
class TeacherForcingPhase:
    name: Literal[PhaseName.TEACHER_FORCING] = PhaseName.TEACHER_FORCING
    epochs: int = 1
    tf_ratio: float = 1.0  # puoi ignorarlo o usarlo in future estensioni


@dataclass(frozen=True)
class MaskedModelingPhase:
    name: Literal[PhaseName.MASKED_MODELING] = PhaseName.MASKED_MODELING
    epochs: int = 1
    mask_prob: float = 0.15


@dataclass(frozen=True)
class ScheduledSamplingPhase:
    name: Literal[PhaseName.SCHEDULED_SAMPLING] = PhaseName.SCHEDULED_SAMPLING
    epochs: int = 1
    tf_ratio_start: float = 1.0
    tf_ratio_end: float = 0.2


PhaseConfig = Union[TeacherForcingPhase, MaskedModelingPhase, ScheduledSamplingPhase]
