from dataclasses import dataclass
from typing import  Union

from .transformer_config import TransformerConfig
from .compile_config import CompileConfig
from .rnn_config import RNNConfig
from .inference_config import InferenceConfig

from ..enums.model_type import ModelType
from ..enums.model_variant import ModelVariant       
from ..enums.decoder_mode import DecoderMode    

@dataclass(frozen=True)
class ModelConfig:
    name: str 
    save_model: bool
    type: ModelType             
    variant: ModelVariant
    decoder_mode: DecoderMode
    compile: CompileConfig 
    inference: InferenceConfig
    params: Union[RNNConfig, TransformerConfig] 