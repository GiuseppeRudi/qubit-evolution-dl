from abc import ABC, abstractmethod
from typing import List
import numpy as np
import tensorflow as tf

class TrainingStrategy(ABC):

    #TODO change the name of this function
    @abstractmethod
    def prepare_inputs(self, X, Y, epoch, total_epochs) ->tuple[list[np.ndarray], np.ndarray]:
        """Prepare the input for the current batch"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def next_dec_input(self,*,y_true_t: tf.Tensor,   # (N,1,D)
        y_pred_t: tf.Tensor,   # (N,1,D)
        epoch:  tf.Tensor ,
        total_epochs: tf.Tensor) -> tf.Tensor:
        """Return decoder input for next timestep (N,1,D)."""
        raise NotImplementedError