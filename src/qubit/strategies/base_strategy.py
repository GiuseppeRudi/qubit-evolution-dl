from abc import ABC, abstractmethod
from typing import List
import numpy as np
import tensorflow as tf

class TrainingStrategy(ABC):

    @abstractmethod
    def prepare_inputs_full_seq(self, X, Y, epoch, total_epochs,horizon) ->tuple[list[np.ndarray], np.ndarray]:
        """Prepare the input for the current batch"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def prepare_inputs_step_wise(self,*,y_true_t: tf.Tensor,   # (N,1,D)
        y_pred_t: tf.Tensor,   # (N,1,D)
        epoch:  tf.Tensor ,
        total_epochs: tf.Tensor) -> tf.Tensor:
        """Return decoder input for next timestep (N,1,D)."""
        raise NotImplementedError