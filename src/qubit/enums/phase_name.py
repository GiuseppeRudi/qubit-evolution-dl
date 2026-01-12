from enum import Enum

class PhaseName(str, Enum):
    TEACHER_FORCING = "teacher_forcing"
    MASKED_MODELING = "masked_modeling"
    SCHEDULED_SAMPLING = "scheduled_sampling"
    FULL_AUTOREGRESSIVE = "full_autoregressive"