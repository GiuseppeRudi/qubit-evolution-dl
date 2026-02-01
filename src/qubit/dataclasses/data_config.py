from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any, Mapping

from ..enums.loss_on import LossOn

from ..utils.config_keys import DATA, DATASET, WINDOWING, DATA_SPLIT,SUPER_RESOLUTION, TASK

from ..enums.task_mode import TaskMode
@dataclass(frozen=True)
class DatasetConfig:
    csv_path: str
    total_qubits: int
    used_qubits: int
    time_steps: int
    n_traj: int
    traj_fraction: float

@dataclass(frozen=True)
class WindowingConfig:
    input_seq_len: int
    output_seq_len: int
    stride: int


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    val_ratio: float
    test_ratio: float

@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetConfig
    windowing: WindowingConfig
    split: SplitConfig

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DataConfig:
        if DATA in d:
            d = d[DATA]
        return DataConfig(
            dataset=DatasetConfig(**d[DATASET]),
            windowing=WindowingConfig(**d[WINDOWING]),
            split=SplitConfig(**d[DATA_SPLIT]),
        )



