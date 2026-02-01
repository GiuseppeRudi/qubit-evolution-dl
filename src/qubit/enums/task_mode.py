from enum import Enum

class TaskMode(str, Enum):
    FORECASTING = "forecasting"
    SR = "super_resolution"