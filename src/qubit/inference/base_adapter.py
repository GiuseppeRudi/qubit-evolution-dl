from __future__ import annotations
from typing import Any

import keras
import tensorflow as tf
from ..enums.start_mode import StartMode
from ..enums.inference_mode import InferenceMode
from ..enums.start_mode import StartMode
from abc import ABC, abstractmethod

from ..models.rnn.lstm2_layer_state import LSTM2LayerTFState 
from ..enums.start_mode import StartMode
from ..enums.inference_mode import InferenceMode

class BaseAutoregressiveAdapter(keras.Model, ABC):
    def __init__(self, *, out_steps: int, start_mode: StartMode, inference_mode: InferenceMode, name: str = "ar_infer"):
        super().__init__(name=name)
        self.out_steps = out_steps
        self.start_mode = start_mode
        self.inference_mode = inference_mode  # FREE_RUNNING / TEACHER_FORCING


    @abstractmethod
    def call(self, inputs, training: bool = False) -> tf.Tensor: ...