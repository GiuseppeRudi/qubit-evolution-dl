from ..model.phase_config import *
from .base_strategy import TrainingStrategy
from .teacher_forcing import TeacherForcingStrategy
from .masked_modelling import MaskedModelingStrategy
from .scheduled_sampling import ScheduledSamplingStrategy

def create_strategy(phase_config: PhaseConfig) -> TrainingStrategy:
    if isinstance(phase_config, TeacherForcingPhase):
        return TeacherForcingStrategy()
    
    elif isinstance(phase_config, ScheduledSamplingPhase):
        return ScheduledSamplingStrategy(
            phase_config.tf_ratio_start,
            phase_config.tf_ratio_end
        )
    
    elif isinstance(phase_config, MaskedModelingPhase):
        return MaskedModelingStrategy(phase_config.mask_prob)
    
    else: raise ValueError(f"Unknown phase config type: {type(phase_config)}")
