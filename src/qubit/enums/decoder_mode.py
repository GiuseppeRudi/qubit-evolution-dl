from enum import Enum

class DecoderMode(str, Enum):
    FULL_SEQ = "FULL_SEQ"   
    STEP_WISE = "STEP_WISE"      
    HYBRID = "HYBRID"      
