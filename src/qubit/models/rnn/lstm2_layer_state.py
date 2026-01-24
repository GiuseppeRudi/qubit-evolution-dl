import tensorflow as tf 
from typing import NamedTuple

class LSTM2LayerTFState(NamedTuple):
    """(h1,c1,h2,c2) each: (batch_size, latent_dim)"""
    h1: tf.Tensor
    c1: tf.Tensor
    h2: tf.Tensor
    c2: tf.Tensor

