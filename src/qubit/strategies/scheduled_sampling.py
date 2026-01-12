from typing import cast
from .base_strategy import TrainingStrategy
from .decoder_utils import make_decoder_inputs
import numpy as np
import tensorflow as tf

class ScheduledSamplingStrategy(TrainingStrategy):
    def __init__(self, tf_ratio_start: float, tf_ratio_end: float):
        self.tf_ratio_start = tf_ratio_start
        self.tf_ratio_end = tf_ratio_end
    
    def prepare_inputs(self, X, Y, epoch, total_epochs,horizon):
        pass
    
    def get_name(self) -> str:
        return f"ScheduledSampling({self.tf_ratio_start} to {self.tf_ratio_end})"
    
    def _tf_ratio(self, epoch: int | tf.Tensor, total_epochs: int | tf.Tensor) -> tf.Tensor:
        # TODO currently is a linear decay but in future introduce different type of decay

        # we put total_epoch -1 because epoch is 0-based
        den = tf.maximum(tf.cast(total_epochs, tf.int32) - 1, 1)
        alpha = tf.math.divide(tf.cast(epoch, tf.float32), tf.cast(den, tf.float32))
        start = tf.cast(self.tf_ratio_start, tf.float32)
        end = tf.cast(self.tf_ratio_end, tf.float32)
        return start + (end - start) * alpha

    def next_dec_input(self, *, y_true_t, y_pred_t, epoch, total_epochs):
        ratio = self._tf_ratio(epoch, total_epochs)  # scalar
        batch_size = tf.shape(y_true_t)[0]  
        feature_dim = tf.shape(y_true_t)[2]  

        # class SS use the broadcast for the feature 
        # instead we also use feature_dim for different behaviour for each feature_dim
        # shape = cast(tf.Tensor, tf.stack([batch_size, 1, feature_dim]))
        # TODO permit to choose this behaviour from the yaml file

        shape = cast(tf.Tensor, tf.stack([batch_size, 1, 1]))
        
        use_teacher = tf.random.uniform(shape, 0, 1.0) < ratio

        # IMPORTANT: stop_gradient sul feedback per evitare grafi enormi/instabilità
        y_pred_in = tf.stop_gradient(y_pred_t)
        return tf.where(use_teacher, y_true_t, y_pred_in)
