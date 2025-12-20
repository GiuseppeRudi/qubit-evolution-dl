from typing import Callable, Dict, Tuple

# builder signature: (x_train, y_train, model_cfg) -> keras.Model
MODEL_REGISTRY: Dict[Tuple[str, str], Callable] = {}

# decorator
# choose the specific function for different type of model and variant
def register_model(model_type: str, variant: str):
    def deco(fn):
        MODEL_REGISTRY[(model_type.upper(), variant.upper())] = fn
        return fn
    return deco

# take the type of neural network and return a specific builder function
def get_builder(model_type: str, variant: str):
    key = (model_type.upper(), variant.upper())
    if key not in MODEL_REGISTRY:
        raise ValueError(f"No builder registered for {key}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]


# Trainer registry
TRAINER_REGISTRY: Dict[str, type] = {}

def register_trainer(strategy: str):
    def deco(cls):
        TRAINER_REGISTRY[strategy.lower()] = cls
        return cls
    return deco

def get_trainer(strategy: str):
    key = strategy.lower()
    if key not in TRAINER_REGISTRY:
        raise ValueError(f"No trainer registered for '{strategy}'. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[key]
