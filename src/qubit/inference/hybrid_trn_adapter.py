from __future__ import annotations
from ..enums.inference_mode import InferenceMode
import tensorflow as tf
from .base_adapter import BaseAutoregressiveAdapter  

class HybridTrnAdapter(BaseAutoregressiveAdapter):
    def __init__(self, model, *, out_steps, feature_dim, inference_mode):
        super().__init__(out_steps=out_steps, inference_mode=inference_mode, name="hybrid_trn_adapter")
        self.model = model
        self.feature_dim = feature_dim

    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:
        return self.model._init_dec0(X)
    
    def encode(self, X: tf.Tensor, training : bool) -> tf.Tensor:
        return self.model._encode(X, training=training)

    def step(self, dec_prefix: tf.Tensor, memory: tf.Tensor, training: bool) -> tf.Tensor:
        return self.model._decode_prefix(dec_prefix, memory, training = training)
        
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

        memory = self.encode(X,training=training)
        # memory.shape(batch_size, input_seq_len, dim_model)

        dec0 = self._init_dec0(X) 
        # dec0 where t = 0  startMode = ZEROS or LAST_x
        # dec0.shape(batch_size, 1 , feature_dim) 

        
        if self.inference_mode == InferenceMode.TEACHER_FORCING and Y_true is not None:
            # Y_true.shape(batch_size, ouput_seq_len , feature_dim)
            
            dec_in = Y_true[:, :self.out_steps, :] # type: ignore[]
            # dec_in.shape(batch_size, out_steps, feature_dim)

            # decoder model function
            # step calls _decoder_prefix that returns the total predictions
            Y_pred = self.step(dec_in, memory, training=training)  
            
            # Y_pred.(batch_size, outsteps, feature_dim)
            return Y_pred
        

        # ! This lines of code below needed only if the inference mode is Free-running 

        # Prefix growing loop: dec_prefix starts from dec0 and grows 1..T_target
        ta = tf.TensorArray(
            dtype=X.dtype,
            size=self.out_steps,
        )

        t0 = tf.constant(0, tf.int32)
        dec_prefix0 = dec0 
        # dec_prefix0.shape(batch_size, 1, feature_dim)

        def cond(t, dec_prefix, ta):
            return t < self.out_steps

        def body(t, dec_prefix, ta):
            
            # Decode the whole prefix and take only the last timestep as current prediction
            y_pred_seq = self.step(dec_prefix, memory, training=training)  
            # y_pred_seq.shape(batch_size, L, feature_dim) where L = t+1
            y_pred_t = y_pred_seq[:, -1:, :]
            # y_pred_t.shape(batch_size, 1, feature_dim)

            y_pred_t_2d = tf.squeeze(y_pred_t, axis=1)
            # y_pred_t_2d,shape(batch_size, feature_dim)
            
            ta = ta.write(t, y_pred_t_2d) 
            # write at the index t the tensor y_pred_t_2d

            # previous => dec_prefix.shape(batch_size, t+1 = L, feature_dim)
            dec_prefix = tf.concat([dec_prefix, y_pred_t], axis=1) 
            # after =>    dec_prefix.shape(batch_size,(t+1)+1 == L +1 ,feature_dim)         

            return t + 1, dec_prefix, ta

        _, _, ta = tf.while_loop(
            cond,
            body,
            loop_vars=[t0, dec_prefix0, ta],
            parallel_iterations=1,
            shape_invariants=[
                t0.get_shape(),
                tf.TensorShape([None, None, self.feature_dim]),  # prefix length grows
                tf.TensorShape([]),
            ],
        )

        # ta.shape(self.outsteps, batch_size, feature_dim)
        Y_pred = tf.transpose(ta.stack(), [1, 0, 2])  
        # Y_pred.shape(batch_size, outsteps , feature_dim)
        return Y_pred


