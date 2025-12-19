from dataclasses import dataclass, field
from typing import Literal, Union

from .training_config import TrainingConfig
from .transformer_config import TransformerConfig
from .compile_config import CompileConfig
from .rnn_config import RNNConfig


ModelType =  str              # libero per adesso piu in la si cambia
ModelVariant = str                            # per ora libera (o puoi fare Literal più avanti)

@dataclass(frozen=True)
class ModelConfig:
    # label libero (nome esperimento/config)
    name: str = "experiment"

    # discriminanti (servono al loader per decidere params/builder)
    type: ModelType = "RNN"                    # "RNN" oppure "TRN"
    variant: ModelVariant = "Seq2Seq"          # es: "Seq2Seq", "Simple", "EncoderOnly"...

    # blocchi comuni
    compile: CompileConfig = field(default_factory=CompileConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # parametri specifici (decisi da type+variant nel loader)
    params: Union[RNNConfig, TransformerConfig] = field(default_factory=RNNConfig)