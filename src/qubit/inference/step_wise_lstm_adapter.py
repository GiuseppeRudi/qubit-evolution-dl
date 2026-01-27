from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from ..enums.start_mode import StartMode

from .base_adapter import BaseAutoregressiveAdapter
from ..models.rnn.step_wise_lstm_model import LSTM2LayerTFState

from ..enums.verbose_mode import VerboseMode


class StepWiseLstmAdapter(BaseAutoregressiveAdapter):
    def __init__(self, model, *, out_steps, inference_mode):
        super().__init__(out_steps=out_steps, inference_mode=inference_mode, name="stepwise_adapter")
        self.model = model

    def encode(self, X: tf.Tensor):
        return self.model._encode(X)

    def step(self, dec_t: tf.Tensor, state):
        return self.model._decode_step(dec_t, state)
    
    def _init_dec0(self, X: tf.Tensor):
        return self.model._init_dec0(X)
