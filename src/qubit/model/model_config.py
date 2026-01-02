from dataclasses import dataclass
from typing import  Union

from .training_config import TrainingConfig
from .transformer_config import TransformerConfig
from .compile_config import CompileConfig
from .rnn_config import RNNConfig
from .inference_config import InferenceConfig

ModelType =  str              
ModelVariant = str                           

@dataclass(frozen=True)
class ModelConfig:
    name: str 
    save_model: bool
    type: ModelType             
    variant: ModelVariant        
    compile: CompileConfig 
    inference: InferenceConfig 
    params: Union[RNNConfig, TransformerConfig] 