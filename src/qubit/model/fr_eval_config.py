from  dataclasses  import dataclass
from typing import Optional

from ..enums.split_name import SplitName

@dataclass(frozen=True)
class FrEvalConfig:
    enabled: bool 
    split: SplitName
    p_eval: float
    every_epochs: int         
