from dataclasses import dataclass

from typing import List
from .phase_config import PhaseConfig
from .fr_eval_config import FrEvalConfig
from ..enums.verbose_mode import VerboseMode
from ..enums.prediction_mode import PredictionMode

@dataclass(frozen=True)
class TrainingConfig:
    prediction_mode: PredictionMode
    epochs: int 
    batch_size: int
    curriculum: List[float] # List[int]
    verbose : VerboseMode
    phases: List[PhaseConfig]
    fr_eval: FrEvalConfig 