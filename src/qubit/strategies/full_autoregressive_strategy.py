from typing import cast
from .base_strategy import TrainingStrategy
from .decoder_utils import make_decoder_inputs
import numpy as np
import tensorflow as tf 


class FullAutoregressiveStrategy(TrainingStrategy):

    def __init__(self, gradient_through_time: bool):
            self.gradient_through_time = gradient_through_time

    def prepare_inputs_full_seq(self, X, Y, epoch, total_epochs, horizon):
        pass
    
    def get_name(self) -> str:
        return f"FullAutoregressive"

    def prepare_inputs_step_wise(self, *, y_true_t, y_pred_t, epoch, total_epochs):
        
        dec_next = y_pred_t
        if not self.gradient_through_time:
            dec_next = tf.stop_gradient(dec_next)
        return dec_next
