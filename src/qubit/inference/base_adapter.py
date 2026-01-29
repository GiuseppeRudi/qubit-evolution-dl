from __future__ import annotations
from typing import Any

import keras
import tensorflow as tf
from abc import ABC, abstractmethod

from ..models.rnn.lstm2_layer_state import LSTM2LayerTFState 
from ..enums.inference_mode import InferenceMode

class BaseAutoregressiveAdapter(keras.Model, ABC):
    def __init__(self, *, out_steps: int, inference_mode: InferenceMode, name: str = "ar_infer"):
        super().__init__(name=name)
        self.out_steps = out_steps
        self.inference_mode = inference_mode  # FREE_RUNNING / TEACHER_FORCING

    @abstractmethod
    def encode(self, X: tf.Tensor) : ...

    @abstractmethod
    def step(self, dec_t: tf.Tensor, state): ...

    @abstractmethod
    def _init_dec0(self, X: tf.Tensor): ...

    @abstractmethod
    def call(self, inputs, training: bool = False): ...
    