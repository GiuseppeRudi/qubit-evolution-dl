from enum import Enum

class MaskScope(str, Enum):
    ELEMENT = "element"   
    FEATURE = "feature"    
    TIME =   "time"
