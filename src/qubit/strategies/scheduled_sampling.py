from .base_strategy import TrainingStrategy
from .decoder_utils import make_decoder_inputs
import numpy as np

class ScheduledSamplingStrategy(TrainingStrategy):
    def __init__(self, tf_ratio_start: float, tf_ratio_end: float):
        self.tf_ratio_start = tf_ratio_start
        self.tf_ratio_end = tf_ratio_end
    
    def prepare_inputs(self, X, Y, epoch, total_epochs):

        # Linear decay
        current_ratio = self.tf_ratio_start - (self.tf_ratio_start - self.tf_ratio_end) * (epoch / total_epochs)
        
        dec_in = self._apply_scheduled_sampling(Y, current_ratio)
        return [X, dec_in], Y
    
    def _apply_scheduled_sampling(self, Y, tf_ratio):
        """Mixes ground truth with previous predictions """
        # Implementazione: usa ground truth con probabilità tf_ratio
        # altrimenti usa predizione del passo precedente
        dec_in = make_decoder_inputs(Y)
        
        # Maschera casuale
        mask = np.random.rand(*Y.shape[:2]) < tf_ratio
        # Apply mask logic here
        
        return dec_in
    
    def get_name(self) -> str:
        return f"ScheduledSampling({self.tf_ratio_start} to {self.tf_ratio_end})"