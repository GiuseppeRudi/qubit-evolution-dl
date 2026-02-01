from enum import Enum

class ModelVariant(str,Enum):
    FORECASTING = "FC"
    SUPER_RESOLUTION = "SR"