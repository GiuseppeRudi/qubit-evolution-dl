from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32