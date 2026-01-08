from typing import cast
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
        # y_true_t.shape = (batch_size , timesteps, feature_dim) where timesteps = 1

        batch_size = tf.shape(y_true_t)[0] 
        feature_dim = tf.shape(y_true_t)[2]

        shape = cast(tf.Tensor, tf.stack([batch_size, 1, feature_dim]))
        
        # TODO currently we apply the same mask for all feature dim  in the feature we modify a different mask for each feature dim 
        # mask is a tensor of boolean with shape (batch_size, 1 ,1 ), min_max_value [0, 1)  
        mask = tf.random.uniform(shape, 0, 1.0) < self.mask_prob

        # (B,1,1) = same mask for all feature dim 
        # (B,1,feature_dim) = different mask for each feature 
        # (B,timesteps,1) = different mask for each timestep
        # (B,timesteps,feature_dim) = different mask for each (timestep, feature)
        # automatic broadcast when we use the same mask for different feature dim in the tf.where function

        # tf.where(condition,x,y) if condition=True take x instead of y
        # TODO currently  use 0 like but in the future we can possibily to choose from yaml file
        return tf.where(mask, tf.zeros_like(y_true_t), y_true_t)

