from  dataclasses  import dataclass
from typing import Optional

from ..enums.split_name import SplitName

@dataclass(frozen=True)
class FrEvalConfig:
    enabled: bool = False
    split: SplitName = SplitName.VAL
    p_eval: Optional[float] = None      # percentuale (es. 20) oppure None
    every_epochs: int = 1               # ogni quante epoche calcolare fr_loss
