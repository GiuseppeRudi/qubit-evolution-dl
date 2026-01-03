from  dataclasses  import dataclass

from ..enums.inference_mode import InferenceMode
from ..enums.start_mode import StartMode
from ..enums.verbose_mode import VerboseMode


@dataclass(frozen=True)
class InferenceConfig:
    mode: InferenceMode 
    start_mode: StartMode 
    verbose : VerboseMode