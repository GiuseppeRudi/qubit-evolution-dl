from dataclasses import dataclass

from typing import List
from .phase_config import PhaseConfig
from .fr_eval_config import FrEvalConfig
from ..enums.verbose_mode import VerboseMode

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int 
    batch_size: int
    approach: str
    verbose : VerboseMode
    phases: List[PhaseConfig]
    fr_eval: FrEvalConfig 