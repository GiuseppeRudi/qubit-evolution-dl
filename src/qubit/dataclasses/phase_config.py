from dataclasses import dataclass
from typing import Literal, Union, Optional

from ..enums.phase_name import PhaseName
from ..enums.mask_mode import MaskMode
from ..enums.mask_scope import MaskScope
from ..enums.ratio_mode import RatioMode

@dataclass(frozen=True)
class TeacherForcingPhase:
    name: Literal[PhaseName.TEACHER_FORCING] 
    epochs: int 
    learning_rate: Optional[float] = None
    clip_norm: Optional[float] = None


@dataclass(frozen=True)
class MaskedModelingPhase:
    name: Literal[PhaseName.MASKED_MODELING] 
    epochs: int 
    mask_prob: float 
    mask_scope: MaskScope
    mask_mode: MaskMode
    mask_value: Optional[float] = None 
    noise_sigma: Optional[float] = None
    noise_replace: Optional[bool] = None
    learning_rate: Optional[float] = None
    clip_norm: Optional[float] = None 

@dataclass(frozen=True)
class ScheduledSamplingPhase:
    name: Literal[PhaseName.SCHEDULED_SAMPLING] 
    epochs: int 
    tf_ratio_start: float 
    tf_ratio_end: float 
    per_feature: bool
    stop_grad_pred: bool
    ratio_mode : RatioMode
    
    # Only if ratio_mode == sigmoid
    mid_point : Optional[float] = None
    sharpness : Optional[float] = None

    # Only if ratio_mode = power
    power_value : Optional[float]= None

    learning_rate: Optional[float] = None
    clip_norm: Optional[float] = None

@dataclass(frozen=True)
class FullAutoregressivePhase:
    name: Literal[PhaseName.FULL_AUTOREGRESSIVE] 
    epochs: int 
    gradient_through_time: bool
    learning_rate: Optional[float] = None
    clip_norm: Optional[float] = None


PhaseConfig = Union[TeacherForcingPhase, MaskedModelingPhase, ScheduledSamplingPhase, FullAutoregressivePhase]

