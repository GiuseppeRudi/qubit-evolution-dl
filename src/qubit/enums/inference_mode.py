from enum import Enum

class InferenceMode(str, Enum):
    TEACHER_FORCING = "teacher_forcing"
    FREE_RUNNING = "free_running"