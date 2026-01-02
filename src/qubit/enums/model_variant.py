from enum import Enum

class ModelVariant(str,Enum):
    SEQ2SEQ = "SEQ2SEQ"
    ENCODERDENSE = "ENCODERDENSE"