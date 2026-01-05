from abc import ABC, abstractmethod
from typing import List
import numpy as np


class TrainingStrategy(ABC):

    
    @abstractmethod
    def prepare_inputs(self, X, Y, epoch, total_epochs) ->tuple[list[np.ndarray], np.ndarray]:
        """Prepare the input for the current batch"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass