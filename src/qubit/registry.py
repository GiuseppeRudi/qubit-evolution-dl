from typing import Callable, Dict, Tuple

from  .enums.model_type import ModelType
from  .enums.model_variant import ModelVariant
from  .enums.decoder_mode import DecoderMode


# builder signature: (x_train, y_train, model_cfg) -> keras.Model
MODEL_REGISTRY: Dict[Tuple[str, str, str], Callable] = {}

# decorator
# choose the specific function for different type of model and variant
def register_model(model_type: ModelType, variant: ModelVariant, decoder_mode : DecoderMode ):
    def deco(fn):
        MODEL_REGISTRY[(model_type.value.upper(), variant.value.upper(), decoder_mode.value.upper())] = fn
        return fn
    return deco

# take the type of neural network and return a specific builder function
def get_builder(model_type: ModelType, variant: ModelVariant, decoder_mode : DecoderMode):
    key = (model_type.value.upper(), variant.value.upper(), decoder_mode.value.upper())
    if key not in MODEL_REGISTRY:
        raise ValueError(f"No builder registered for {key}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]


# Trainer registry
TRAINER_REGISTRY: Dict[str, type] = {}

def register_trainer(type: ModelType):
    def deco(cls):
        TRAINER_REGISTRY[type.value.lower()] = cls
        return cls
    return deco

def get_trainer(type: ModelType):
    key = type.value.lower()
    print(list(TRAINER_REGISTRY.keys()))
    if key not in TRAINER_REGISTRY:
        raise ValueError(f"No trainer registered for '{type}'. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[key]
