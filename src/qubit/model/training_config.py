from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int 
    batch_size: int 