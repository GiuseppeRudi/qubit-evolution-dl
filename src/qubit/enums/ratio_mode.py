from enum import Enum

class RatioMode(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"
    POWER = "power"
