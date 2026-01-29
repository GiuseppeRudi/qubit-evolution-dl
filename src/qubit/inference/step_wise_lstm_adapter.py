from __future__ import annotations
import tensorflow as tf

from .base_adapter import BaseAutoregressiveAdapter
from ..models.rnn.step_wise_lstm_model import LSTM2LayerTFState
from ..enums.inference_mode import InferenceMode

class StepWiseLstmAdapter(BaseAutoregressiveAdapter):
    def __init__(self, model, *, out_steps, inference_mode):
        super().__init__(out_steps=out_steps, inference_mode=inference_mode, name="stepwise_adapter")
        self.model = model

    def encode(self, X: tf.Tensor):
        return self.model._encode(X)

    def step(self, dec_t: tf.Tensor, state):
        return self.model._decode_step(dec_t, state)
    
    def _init_dec0(self, X: tf.Tensor):
        return self.model._init_dec0(X)
    
    def call(self, inputs, training: bool = False) -> tf.Tensor:
        
        # self.outsteps 
        # if prediction_mode == ALL outsteps = output_seq_len
        # if prediction_mode == HORIZON outsteps = max(horizons) for a specific phase (strategy)

        # X.shape(batch_size, input_seq_len, feature_dim)
  
        # if inference_mode == TEACHER_FORCING , inputs = (X, y_true)
        if isinstance(inputs, (tuple, list)):
            # Y_true (batch_size, t, feature_dim)
            # if prediction_mode == ALL t = output_seq_len
            # if prediction_mode == HORIZON t = max(horizons) for a specific phase (strategy)
            X, Y_true = inputs
        
        # if inference_mode == FREE_RUNNING , inputs = X
        else:
            X, Y_true = inputs, None


        state : LSTM2LayerTFState = self.encode(X)
        # LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)
        # h1 and c1 from encoder layer 1 
        # h2 and c2 from encoder layer 2 

        # h* and c* shape(batch_size, latent_dim)
         
        dec_t = self._init_dec0(X)
        # dec_t where t = 0  startMode = ZEROS or LAST_x
        # dec_t.shape(batch_size, 1 , feature_dim) 

        ta = tf.TensorArray(dtype=X.dtype, 
                            size=self.out_steps)
        
        use_tf = (self.inference_mode == InferenceMode.TEACHER_FORCING) and (Y_true is not None)
        
        # ta.shape(element_size = out_steps, element_shape = (batch_size, feature_dim))
        
        def cond(t, dec_t, state, ta):
            return t < self.out_steps
        
        def body(t, dec_t, state, ta):
            
            # dec_t.shape(batch_size, 1, feature_dim)
            # if t != 0 dec_t => previous prediction, instead the start mode

            # if t = 0 hidden states and cell states from the encoder
            # if t > 0 hidden states and cell states from the decoder at timestep t-1
            
            # state => LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)
            # h* and c* are shape(batch_size, latent_dim)

            y_pred_t, state = self.step(dec_t, state)  
            # y_pred_t.shape(batch_size, 1, feature_dim)

            y_pred_t_2d = tf.squeeze(y_pred_t, axis=1)  
            # y_pred_t_2d.shape(batch_size, feature_dim)
            # state needed at the timesteps t+1 from decoder input for the next predictions 
            
            ta = ta.write(t, y_pred_t_2d)
            # write at the index t the tensor y_pred_t_2d

            # dec_t will be the decoder input for the next iteration (t+1) 
            
            if use_tf:
                dec_t = Y_true[:, t:t+1, :] #type: ignore[] 
            else:
                dec_t = y_pred_t

            return t + 1, dec_t, state, ta

        # start from t = 0 and stop where t < outsteps
        t0 = tf.constant(0, tf.int32)

        _, _, state, ta = tf.while_loop(
            cond,
            body,
            (t0, dec_t, state, ta),
        )

        y_pred = ta.stack()                 
        # y_pred.shape(element_size = out_steps, batch_size, feature_dim)

        # element_size will be the output_seq_len 
        # stack: (element_size = outsteps, batch_size, feature_dim) -> transpose: (batch_size, outsteps, feature_dim)
        y_pred = tf.transpose(y_pred, [1, 0, 2])  

        return y_pred

