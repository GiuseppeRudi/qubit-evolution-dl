from typing import cast
from .base_strategy import TrainingStrategy
from .decoder_utils import make_decoder_inputs
import numpy as np
import tensorflow as tf 


from ..enums.mask_mode import MaskMode
from ..enums.mask_scope import MaskScope

class MaskedModelingStrategy(TrainingStrategy):
    def __init__(self, mask_prob: float, mask_mode : MaskMode , mask_scope : MaskScope , mask_value : float  , noise_sigma : float):
        self.mask_prob = mask_prob
        self.mask_mode = mask_mode
        self.mask_scope = mask_scope
        self.mask_value = mask_value
        self.noise_sigma = noise_sigma
    
    def prepare_inputs(self, X, Y, epoch, total_epochs, horizon):
        
        # decoder input with teacher forcing = ground truth 
        dec_in = make_decoder_inputs(Y, horizon)

        # denoising = to dirty the input 
        dec_in_masked = self._apply_masking_np(dec_in)

        return [X, dec_in_masked], Y
    

    def _apply_masking_np(self, dec_in: np.ndarray) -> np.ndarray:
        # dec_in: (N,T,F)
        N, T, F = dec_in.shape

        # build mask depending on scope
        if self.mask_scope == MaskScope.TIME:
            # mask per timestep (N,T,1) broadcast su F
            m = (np.random.rand(N, T, 1) < self.mask_prob)
        elif self.mask_scope == MaskScope.FEATURE:
            # mask per feature (N,1,F) broadcast su T
            m = (np.random.rand(N, 1, F) < self.mask_prob)
        else:  # ELEMENT
            m = (np.random.rand(N, T, F) < self.mask_prob)

        masked = dec_in.copy()

        # choose replacement based on mode
        if self.mask_mode == MaskMode.ZERO:
            repl = 0.0
            masked[m] = repl

        elif self.mask_mode == MaskMode.CONSTANT:
            masked[m] = self.mask_value

        elif self.mask_mode == MaskMode.NOISE:
            noise = np.random.normal(loc=0.0, scale=self.noise_sigma, size=dec_in.shape).astype(dec_in.dtype)
            # qui: sostituisci con rumore “puro”
            # alternativa: dec_in + noise (corruzione, non mascheramento)
            masked[m] = noise[m]

        else:
            raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

        return masked

        
    # def _apply_masking(self, dec_in):
    #     # dec_in.shape(N,T,F) = (num_trajectories_windowed , timesteps, feature_dim)

    #     # with rand create a matrix with casual number from [0,1)
    #     # obtain a boolean matrix take in only (N,T)
    #     # True ~ mask_prob%
    #     mask = np.random.rand(*dec_in.shape[:2]) < self.mask_prob
        
    #     masked = dec_in.copy()
        
    #     # masked[n,t,:] this is the same thing where (n,t) = mask 
    #     masked[mask] = 0  # TODO or different special token [MASK]

    #     return masked
    
    def _make_mask_tf(self, x: tf.Tensor) -> tf.Tensor:
        # x shape: (B,T,F)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        F = tf.shape(x)[2]

        if self.mask_scope == MaskScope.TIME:
            shape = tf.stack([B, T, 1])
        elif self.mask_scope == MaskScope.FEATURE:
            shape = tf.stack([B, 1, F])
        else:  # ELEMENT
            shape = tf.stack([B, T, F])

        m = tf.random.uniform(shape, 0.0, 1.0) < self.mask_prob
        # broadcast automatico quando m ha 1 in una dimensione
        return m


    def _apply_masking_tf(self, x: tf.Tensor, *, mask: tf.Tensor) -> tf.Tensor:
        # x shape (B,T,F) ; mask bool broadcastable to x
        if self.mask_mode == MaskMode.ZERO:
            repl = tf.zeros_like(x)
            return tf.where(mask, repl, x)

        if self.mask_mode == MaskMode.CONSTANT:
            repl = tf.ones_like(x) * tf.cast(self.mask_value, x.dtype)
            return tf.where(mask, repl, x)

        if self.mask_mode == MaskMode.NOISE:
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_sigma, dtype=x.dtype)
            # sostituisci con rumore
            return tf.where(mask, noise, x)
            # alternativa (spesso migliore): corrompi invece di sostituire
            # return tf.where(mask, x + noise, x)

        raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

    def get_name(self) -> str:
            return f"MaskedModeling(p={self.mask_prob})"
        
    def next_dec_input(self, *, y_true_t, y_pred_t, epoch, total_epochs):
        # y_true_t: (B,1,F)
        mask = self._make_mask_tf(y_true_t)              # (B,1,F) o (B,1,1) ecc.
        return self._apply_masking_tf(y_true_t, mask=mask)


    # # TODO change the function becuase we want to use a apply mask functions
    # def next_dec_input(self, *, y_true_t, y_pred_t, epoch, total_epochs):
    #     # y_true_t.shape = (batch_size , timesteps, feature_dim) where timesteps = 1

    #     batch_size = tf.shape(y_true_t)[0] 
    #     feature_dim = tf.shape(y_true_t)[2]

    #     shape = cast(tf.Tensor, tf.stack([batch_size, 1, feature_dim]))
        
    #     # TODO currently we apply the same mask for all feature dim  in the feature we modify a different mask for each feature dim 
    #     # mask is a tensor of boolean with shape (batch_size, 1 ,1 ), min_max_value [0, 1)  
    #     mask = tf.random.uniform(shape, 0, 1.0) < self.mask_prob

    #     # (B,1,1) = same mask for all feature dim 
    #     # (B,1,feature_dim) = different mask for each feature 
    #     # (B,timesteps,1) = different mask for each timestep
    #     # (B,timesteps,feature_dim) = different mask for each (timestep, feature)
    #     # automatic broadcast when we use the same mask for different feature dim in the tf.where function

    #     # tf.where(condition,x,y) if condition=True take x instead of y
    #     # TODO currently  use 0 like but in the future we can possibily to choose from yaml file
    #     return tf.where(mask, tf.zeros_like(y_true_t), y_true_t)

