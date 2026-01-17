from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any, Mapping

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
    def from_dict(d: Mapping[str, Any]) -> "DataConfig":
        if "data" in d:
            d = d["data"]
        return DataConfig(
            dataset=DatasetConfig(**d["dataset"]),
            windowing=WindowingConfig(**d["windowing"]),
            split=SplitConfig(**d["split"]),
        )



