from enum import Enum

class MaskMode(str, Enum):
    ZERO = "zero"   
    CONSTANT = "constant"    
    NOISE =   "noise"
