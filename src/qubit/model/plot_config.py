from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PlotConfig:
    sample_index: List[int]
    save_plots: bool 
    save_artifacts: bool 

