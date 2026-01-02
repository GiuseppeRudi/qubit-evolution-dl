from enum import Enum

class StartMode(str, Enum):
    ZEROS = "zeros"
    LAST_X = "last_x"