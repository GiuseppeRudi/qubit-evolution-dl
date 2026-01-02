from dataclasses import dataclass

from typing import List
from .phase_config import PhaseConfig
from .fr_eval_config import FrEvalConfig

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int 
    batch_size: int
    approach: str
    phases: List[PhaseConfig] = None  # type: ignore
    fr_eval: FrEvalConfig = FrEvalConfig()