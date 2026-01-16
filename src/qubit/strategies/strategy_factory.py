from ..model.phase_config import *
from .base_strategy import TrainingStrategy
from .teacher_forcing import TeacherForcingStrategy
from .masked_modelling import MaskedModelingStrategy
from .scheduled_sampling import ScheduledSamplingStrategy
from .full_autoregressive_strategy import FullAutoregressiveStrategy


def create_strategy(phase_config: PhaseConfig) -> TrainingStrategy:
    if isinstance(phase_config, TeacherForcingPhase):
        return TeacherForcingStrategy()
    
    elif isinstance(phase_config, ScheduledSamplingPhase):
        return ScheduledSamplingStrategy(
            phase_config.tf_ratio_start,
            phase_config.tf_ratio_end
        )
    
    elif isinstance(phase_config, MaskedModelingPhase):
        return MaskedModelingStrategy(phase_config.mask_prob, 
                                      phase_config.mask_mode, 
                                      phase_config.mask_scope,
                                      phase_config.mask_value if phase_config.mask_value is not None else 0,
                                      phase_config.noise_sigma  if phase_config.noise_sigma is not None else 0   )
    
    elif isinstance(phase_config, FullAutoregressivePhase):
        return FullAutoregressiveStrategy(phase_config.gradient_through_time)
    
    else: raise ValueError(f"Unknown phase config type: {type(phase_config)}")
