from .base_strategy import TrainingStrategy
from .decoder_utils import make_decoder_inputs
import numpy as np
import tensorflow as tf 

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
    

    # TODO change the function becuase we want to use a apply mask functions
    def next_dec_input(self, *, y_true_t, y_pred_t, epoch, total_epochs):
        # mask per-sample per-timestep (N,1,1) -> broadcast su D
        N = tf.gather(tf.shape(y_true_t), 0)  # invece di tf.shape(y_true_t)[0]
        mask = tf.random.uniform((N, 1, 1), 0, 1.0) < self.mask_prob
        return tf.where(mask, tf.cast(self.mask_prob, y_true_t.dtype) * tf.ones_like(y_true_t), y_true_t)

