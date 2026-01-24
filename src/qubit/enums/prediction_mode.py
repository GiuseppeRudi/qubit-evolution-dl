from enum import Enum

class PredictionMode(str,Enum):
    ALL = "all"
    HORIZON = "horizon"