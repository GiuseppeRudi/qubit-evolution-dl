from .base_strategy import TrainingStrategy
from .decoder_utils import make_decoder_inputs
import numpy as np


class MaskedModelingStrategy(TrainingStrategy):
    def __init__(self, mask_prob: float = 0.15):
        self.mask_prob = mask_prob
    
    def prepare_inputs(self, X, Y, epoch, total_epochs):
        
        # decoder input with teacher forcing = ground truth 
        dec_in = make_decoder_inputs(Y)

        # denoising = to dirty the input 
        dec_in_masked = self._apply_masking(dec_in)

        return [X, dec_in_masked], Y
    
    def _apply_masking(self, dec_in):
        # dec_in.shape(N,T,F) = (num_trajectories_windowed , timesteps, feature_dim)

        # with rand create a matrix with casual number from [0,1)
        # obtain a boolean matrix take in only (N,T)
        # True ~ mask_prob%
        mask = np.random.rand(*dec_in.shape[:2]) < self.mask_prob
        
        masked = dec_in.copy()
        
        # masked[n,t,:] this is the same thing where (n,t) = mask 
        masked[mask] = 0  # TODO or different special token [MASK]

        return masked
    
    def get_name(self) -> str:
        return f"MaskedModeling(p={self.mask_prob})"