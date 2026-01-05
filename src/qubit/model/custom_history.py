from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CustomHistory:
    history: Dict[str, List[float]]
    epoch: List[int]
