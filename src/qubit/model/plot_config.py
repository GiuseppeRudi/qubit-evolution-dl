from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PlotConfig:
    pred_all: bool 
    sample_index: List[int]
    save_plots: bool 
    predictions_dir: Path = Path("predictions")

    def __post_init__(self):
        # dataclass frozen: serve object.__setattr__
        if self.sample_index is None:
            object.__setattr__(self, "sample_index", [0])