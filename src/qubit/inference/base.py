from __future__ import annotations
from typing import Any

import keras
import tensorflow as tf
from ..enums.start_mode import StartMode
from ..enums.inference_mode import InferenceMode
from ..enums.start_mode import StartMode
from abc import ABC, abstractmethod

from ..models.rnn.lstm2_layer_state import LSTM2LayerTFState 
from ..enums.start_mode import StartMode
from ..enums.inference_mode import InferenceMode

class BaseAutoregressiveAdapter(keras.Model, ABC):
    def __init__(self, *, out_steps: int, start_mode: StartMode, inference_mode: InferenceMode, name: str = "ar_infer"):
        super().__init__(name=name)
        self.out_steps = out_steps
        self.start_mode = start_mode
        self.inference_mode = inference_mode  # FREE_RUNNING / TEACHER_FORCING


    def init_decoder_input(self, X: tf.Tensor, *, start_mode: StartMode) -> tf.Tensor:
        
        # X.shape(batch_size, input_seq_len, feature_dim)
        if start_mode == StartMode.LAST_X:
            return X[:, -1:, :]  # (batch_size, 1, feature_dim)
        
        batch_size = tf.shape(X)[0]
        feature_dim = tf.shape(X)[2]

        return tf.zeros((batch_size, 1, feature_dim), dtype=X.dtype)
    

    @abstractmethod
    def encode(self, X: tf.Tensor) -> Any: ...


    @abstractmethod
    def step(self, dec_t: tf.Tensor, state: Any) -> tuple[tf.Tensor, Any]: ...

    @tf.function
    def call(self, inputs, training: bool = False) -> tf.Tensor:
        
        # self.outsteps 
        # if prediction_mode == ALL outsteps = output_seq_len
        # if prediction_mode == HORIZON outsteps = horizon for a specific phase (strategy)

        # X.shape(batch_size, input_seq_len, feature_dim)

        # if inference_mode == TEACHER_FORCING , inputs = (X, y_true)
        if isinstance(inputs, (tuple, list)):
            # y_true (batch_size, t, feature_dim)
            # if prediction_mode == ALL t = output_seq_len
            # if prediction_mode == HORIZON t = horizon for a specific phase (strategy)
            X, y_true = inputs
        
        # if inference_mode == FREE_RUNNING , inputs = X
        else:
            X, y_true = inputs, None

        state : LSTM2LayerTFState = self.encode(X)
        # LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)
        # h1 and c1 from encoder layer 1 
        # h2 and c2 from encoder layer 2 

        # h* and c* shape(batch_size, latent_dim)
         
        dec_t = self.init_decoder_input(X, start_mode=self.start_mode)
        # dec_t where t = 0  startMode = ZEROS or LAST_x
        # dec_t.shape(batch_size, 1 , feature_dim) 

        ta = tf.TensorArray(dtype=X.dtype, size=self.out_steps)
        # ta.shape(element_size = out_steps, batch_size, 1, feature_dim)
        
        def body(t, dec_t, state, ta):
            
            # dec_t.shape(batch_size, 1, feature_dim)
            # if t != 0 dec_t => previous prediction, instead the start mode

            # if t = 0 hidden states and cell states from the encoder
            # if t > 0 hidden states and cell states from the decoder at timestep t-1
            
            # state => LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)
            # h* and c* are shape(batch_size, latent_dim)

            y_t, state = self.step(dec_t, state)    

            # y_t.shape(batch_size, 1 , feature_dim)
            # state needed at the timesteps t+1 from decoder input for the next predictions 
            
            ta = ta.write(t, y_t)
            # write at the index t the tensor y_t

            # dec_t will be the decoder input for the next iteration (t+1) 
            
            if self.inference_mode == InferenceMode.TEACHER_FORCING and y_true is not None:
                dec_t = y_true[:, t:t+1, :]
            else:
                dec_t = y_t

            return t + 1, dec_t, state, ta

        # start from t = 0 and stop where t < outsteps
        t0 = tf.constant(0, tf.int32)

        _, _, state, ta = tf.while_loop(
            lambda t, *_: t < self.out_steps,
            body,
            (t0, dec_t, state, ta),
        )

        y = ta.stack()                 
        # y.shape(element_size = out_steps, batch_size, 1, feature_dim)

        tf.print(tf.shape(y))
        # element_size will be the output_seq_len 
        y = tf.transpose(y, [1, 0, 2, 3])  # (out_steps, batch_size, 1, feature_dim) => (batch_size, out_steps, 1, feature_dim)
        
        y = tf.squeeze(y, axis=2)     
        # y.shape(batch_size, outsteps, feature_dim)
        return y
