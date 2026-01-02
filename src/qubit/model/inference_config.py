from  dataclasses  import dataclass

from ..enums.inference_mode import InferenceMode
from ..enums.start_mode import StartMode

@dataclass(frozen=True)
class InferenceConfig:
    mode: InferenceMode 
    start_mode: StartMode 