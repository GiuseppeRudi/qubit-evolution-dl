from typing import cast
from .base_strategy import TrainingStrategy
from .decoder_utils import make_decoder_inputs
import numpy as np
import tensorflow as tf 


from ..enums.mask_mode import MaskMode
from ..enums.mask_scope import MaskScope

class MaskedModellingStrategy(TrainingStrategy):
    def __init__(self, mask_prob: float, mask_mode : MaskMode , mask_scope : MaskScope , mask_value : float  , noise_sigma : float):
        self.mask_prob = mask_prob
        self.mask_mode = mask_mode
        self.mask_scope = mask_scope
        self.mask_value = mask_value
        self.noise_sigma = noise_sigma


    def prepare_inputs_full_seq(self, X, Y, epoch, total_epochs, horizon):
        # X: (B,Tin,F), Y: (B,Tout,F)  <-- devono essere tf.Tensor se vuoi graph mode

        X = tf.convert_to_tensor(X)   # accetta NumPy direttamente
        Y = tf.convert_to_tensor(Y)

        h = tf.cast(horizon, tf.int32)
        Y_h = Y[:, :h, :]  # (B,h,F)

        # decoder input: shift right (t=0 zeri, poi y_{t-1})
        z0 = tf.zeros_like(Y_h[:, :1, :])                 # (B,1,F)
        dec_in = tf.concat([z0, Y_h[:, :-1, :]], axis=1)  # (B,h,F)

        dec_in_masked = self.apply_mask_tf(dec_in, training=True)
        return [X, dec_in_masked], Y_h
        
    # def prepare_inputs_full_seq(self, X, Y, epoch, total_epochs, horizon):
        
    #     # decoder input with teacher forcing = ground truth 
    #     dec_in = make_decoder_inputs(Y, horizon)

    #     # denoising = to dirty the input 
    #     dec_in_masked = self._apply_masking_np(dec_in)

    #     Y_h = Y[:, :horizon, :]   
    #     return [X, dec_in_masked], Y_h
    
    def _make_mask_tf(self, x: tf.Tensor) -> tf.Tensor:
        # x: (B,T,F)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        F = tf.shape(x)[2]

        if self.mask_scope == MaskScope.TIME:
            shape = tf.stack([B, T, 1])     # broadcast su F
        elif self.mask_scope == MaskScope.FEATURE:
            shape = tf.stack([B, 1, F])     # broadcast su T
        else:  # ELEMENT
            shape = tf.stack([B, T, F])

        return tf.random.uniform(shape, 0.0, 1.0) < self.mask_prob


    def apply_mask_tf(self, x: tf.Tensor, *, training: bool) -> tf.Tensor:
        # se vuoi che in inference non mascheri
        if not training:
            return x

        m = self._make_mask_tf(x)  # broadcastable su x

        if self.mask_mode == MaskMode.ZERO:
            return tf.where(m, tf.zeros_like(x), x)

        if self.mask_mode == MaskMode.CONSTANT:
            repl = tf.cast(self.mask_value, x.dtype)
            return tf.where(m, tf.ones_like(x) * repl, x)

        if self.mask_mode == MaskMode.NOISE:
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_sigma, dtype=x.dtype)
            return tf.where(m, noise, x)
            # alternativa “corruption” spesso migliore:
            # return tf.where(m, x + noise, x)

        raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")


    # def _apply_masking_np(self, dec_in: np.ndarray) -> np.ndarray:
    #     # dec_in: (N,T,F)
    #     N, T, F = dec_in.shape

    #     print(f"dec_in.shape = {dec_in.shape} (_apply_masking_np)")

    #     # build mask depending on scope
    #     if self.mask_scope == MaskScope.TIME:
    #         # mask per timestep (N,T,1) broadcast su F
    #         m = (np.random.rand(N, T, 1) < self.mask_prob)
    #     elif self.mask_scope == MaskScope.FEATURE:
    #         # mask per feature (N,1,F) broadcast su T
    #         m = (np.random.rand(N, 1, F) < self.mask_prob)
    #     else:  # ELEMENT
    #         m = (np.random.rand(N, T,F) < self.mask_prob)

    #     masked = dec_in.copy()
    #     m_full = np.broadcast_to(m, dec_in.shape)  # (N,T,F)

    #     # choose replacement based on mode
    #     if self.mask_mode == MaskMode.ZERO:
    #         masked = dec_in.copy()
    #         masked[m_full] = 0.0


    #     elif self.mask_mode == MaskMode.CONSTANT:
    #         masked[m_full] = self.mask_value


    #     # TODO implement noise injections and thing about if it is correct to implement here or in SS strategy
    #     elif self.mask_mode == MaskMode.NOISE:
    #         noise = np.random.normal(loc=0.0, scale=self.noise_sigma, size=dec_in.shape).astype(dec_in.dtype)
    #         #! change the real noise injections 
    #         #! alternative: dec_in + noise (corruption, not mash)
    #         masked[m_full] = noise[m]

    #     else:
    #         raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")
        
    #     print(f"masked.shape = {masked.shape} (_apply_masking_np)")

    #     return masked

    
    
    # def _make_mask_tf(self, x: tf.Tensor) -> tf.Tensor:
    #     # x shape: (B,T,F)
    #     B = tf.shape(x)[0]
    #     T = tf.shape(x)[1]
    #     F = tf.shape(x)[2]

    #     if self.mask_scope == MaskScope.TIME:
    #         shape = tf.stack([B, T, 1])
    #     elif self.mask_scope == MaskScope.FEATURE:
    #         shape = tf.stack([B, 1, F])
    #     else:  # ELEMENT
    #         shape = tf.stack([B, T, F])

    #     m = tf.random.uniform(shape, 0.0, 1.0) < self.mask_prob
    #     # broadcast automatico quando m ha 1 in una dimensione
    #     return m


    # def _apply_masking_tf(self, y: tf.Tensor, *, mask: tf.Tensor) -> tf.Tensor:
    #     # y shape (B,T,F) ; mask bool broadcastable to y
    #     if self.mask_mode == MaskMode.ZERO:
    #         repl = tf.zeros_like(y)
    #         return tf.where(mask, repl, y)

    #     if self.mask_mode == MaskMode.CONSTANT:
    #         repl = tf.ones_like(y) * tf.cast(self.mask_value, y.dtype)
    #         return tf.where(mask, repl, y)

    #     if self.mask_mode == MaskMode.NOISE:
    #         noise = tf.random.normal(tf.shape(y), mean=0.0, stddev=self.noise_sigma, dtype=y.dtype)
    #         # sostituisci con rumore
    #         return tf.where(mask, noise, y)
    #         # alternativa (spesso migliore): corrompi invece di sostituire
    #         # return tf.where(mask, y + noise, y)

    #     raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

    def get_name(self) -> str:
            return f"MaskedModeling(p={self.mask_prob})"
        
    def prepare_inputs_step_wise(self, *, y_true_t, y_pred_t, epoch, total_epochs):
        # y_true_t: (B,1,F)
        return self.apply_mask_tf(y_true_t, training=True)


