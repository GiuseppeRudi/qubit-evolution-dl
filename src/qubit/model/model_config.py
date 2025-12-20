from dataclasses import dataclass, field
from typing import Literal, Union

from .training_config import TrainingConfig
from .transformer_config import TransformerConfig
from .compile_config import CompileConfig
from .rnn_config import RNNConfig


ModelType =  str              
ModelVariant = str                           

@dataclass(frozen=True)
class ModelConfig:
    name: str 
    type: ModelType             
    variant: ModelVariant        
    compile: CompileConfig 
    training: TrainingConfig 
    params: Union[RNNConfig, TransformerConfig] 