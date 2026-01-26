from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, cast
from ...enums.prediction_mode import PredictionMode
import tensorflow as tf
import keras
from keras import layers

from ...enums.start_mode import StartMode
from ..strategy_chooser import StrategyChooserModel

from .lstm2_layer_state import LSTM2LayerTFState

from ...utils.layers_names import ENC_LSTM_1, ENC_LSTM_2, DEC_LSTM_1, DEC_LSTM_2, OUT_DENSE

class StepWiseLstmModel(StrategyChooserModel):
    def __init__(self, *, feature_dim: int, latent_dim: int, start_mode: StartMode, prediction_mode_id: int, t_out : int):
        super().__init__(t_out=t_out, prediction_mode_id=prediction_mode_id)
        
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim  # dimensions of hidden states and cell states 
        self.start_mode = start_mode  # Zeros or last_x

        # encoder  (2 stacked LSTM)
        self.enc_lstm_1 = layers.LSTM(latent_dim, return_state=True,return_sequences=True, name=ENC_LSTM_1)
        self.enc_lstm_2 = layers.LSTM(latent_dim, return_state=True, name=ENC_LSTM_2)

        # decoder (step-by-step, input (batch_size,1,feature_dim)
        self.dec_lstm_1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name=DEC_LSTM_1)
        self.dec_lstm_2 = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name=DEC_LSTM_2)

        self.out_dense = layers.Dense(feature_dim, name=OUT_DENSE)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")


    @property
    def metrics(self):
        return [self.loss_tracker] + super().metrics

    # this function is useful to create for each layers the dimension of the weights accordly 
    # from the last dimension => feature_dim 
    def build(self, input_shape):
        
        # input_shape: (batch_size, input_seq_len, feature_dim)
        # input shape as parameter is needed to respect the build in function

        # Encoder build
        # we insert only the last dimension because only this is important for the weights
        
        # input => (batch_size, input_seq_len, feature_dim)
        # output => (batch_size, t_in, latent_dim)
        self.enc_lstm_1.build((None, None, self.feature_dim))

        # since enc_lstm_2 takes in input the sequence of enc_lstm1 so the 3rd dimension is latent_dim
        # input => (batch_size, t_in, latent_dim)
        self.enc_lstm_2.build((None, None, self.latent_dim))
        
        # Decoder build

        # decoder take in input the dec_in (previous predictions with possibility to apply a different strategies)
        # at the start dec_in depends on the start_mode (zeros or last x)
        # input => dec_in.shape(batch_size, 1, feature_dim)
        # output => (batch_size, 1, latent_dim)
        self.dec_lstm_1.build((None, 1, self.feature_dim)) 
        
        # input => (batch_size, 1, latent_dim)
        # output => (batch_size, 1, latent_dim)
        self.dec_lstm_2.build((None, 1, self.latent_dim))

        # input => (batch_size, 1, latent_dim)
        # output => (batch_size, 1, feature_dim)
        self.out_dense.build((None, 1, self.latent_dim))

        super().build(input_shape)

    # called when we perform predict in fr_running_callback and we predict always up to output_seq_len
    def call(self, X: tf.Tensor) -> tf.Tensor:
        tf.print("call")
        tf.print(X.shape)

        # X.shape(batch_size, input_seq_len, feature_dim)
        X = tf.ensure_shape(X, [None, None, self.feature_dim])

        T_out = tf.convert_to_tensor(self.rt.t_out)

        ta = tf.TensorArray(dtype=X.dtype, 
                             size=T_out, 
                             element_shape=tf.TensorShape([None, self.feature_dim]))
        tf.print(ta.shape)
        # ta.shape(T_out , batch_size, feature_dim)
        
        t0 = tf.constant(0, tf.int32)

        state = self._encode(X)
        # LSTM2LayerTFState(h1,c1,h2,c2) .shape(batch_size,latent_dim)
        # state are the hidden states and cell states of the two encoder layers 

        # inititialize to zeros or last_x based on the start_mode parameter
        # dec_t.shape(batch_size, 1, feature_dim) where t = 0 
        dec_t = self._init_dec0(X)

        def cond(t, dec_t, state, ta):
            return t < T_out

        def body(t, dec_t, state, ta):
            
            y_pred_t, state = self._decode_step(dec_t, state)  
            # y_pred_t.shape(batch_size, 1, feature_dim)
            y_pred_t_2d = tf.squeeze(y_pred_t, axis=1)
            # y_pred_t_nd.shape(batch_size, feature_dim)

            ta = ta.write(t, y_pred_t_2d)
            # ta.shape(index = t, element.shape(batch_size, feature_dim))

            # free-running: next input = prediction
            dec_t = y_pred_t
            return t + 1, dec_t, state, ta

        _, _, _, ta = tf.while_loop(cond, body, [t0, dec_t, state, ta], parallel_iterations=1)
        
        # (num_elements = T_out , batch_size , feature_dim) => (batch_size, T_out, feature_dim) 
        Y_pred = tf.transpose(ta.stack(), [1, 0, 2])
        
        # Y_pred.shape(batch_size,T_out = output_seq_len , feature_dim) 
        return Y_pred

    def _encode(self, X: tf.Tensor) -> LSTM2LayerTFState:
        x_seq, h1, c1 = self.enc_lstm_1(X)   
        _, h2, c2 = self.enc_lstm_2(x_seq)
        return LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)

    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:
        if self.start_mode == StartMode.LAST_X:
            input_seq_len = tf.shape(X)[1]
            last = tf.gather(X, input_seq_len - 1, axis=1)   # (batch_size, feature_dim)
            return tf.expand_dims(last, axis=1)  # (batch_size, 1, feature_dim)

        batch_size = tf.shape(X)[0]
        feature_dim = self.feature_dim         
        return tf.zeros(tf.stack([batch_size, 1, feature_dim]), dtype=X.dtype)

    def _decode_step(self, dec_t: tf.Tensor, state: LSTM2LayerTFState) -> tuple[tf.Tensor, LSTM2LayerTFState]:
        # dec_t.shape(batch_size, 1, feauture_dim)
        # at t = 0 dec_t based on the start mode is Last_x or Zeros
        # at t > 0 dec_t are the previous predicitions at t-1 
        
        # h* and c* are always with shape(batch_size, latent_dim)
       
        # state
        # at t = 0 are the hidden and cell states from encoder_input
        # at t > 0 are the ones from the previous decoder_output
        
        # input => initial_states(h1,c1 from previous decoder states output or if t = 0 from encoder state outputs)
        # output => dec_seq1.shape(batch_size, 1, latent_dim) 
        dec_seq1, h1_out, c1_out = self.dec_lstm_1(dec_t, initial_state=[state.h1, state.c1])

        # input => initial_states(h1,c1 from previous decoder states output or if t = 0 from encoder state outputs)
        # output => dec_seq2.shape(batch_size, 1, latent_dim) 
        dec_seq2, h2_out, c2_out = self.dec_lstm_2(dec_seq1, initial_state=[state.h2, state.c2])
        
        # input => dec_seq2.shape(batch_size, 1, latent_dim)
        # output => y_pred_t.shape(batch_size, 1, feature_dim)
        y_pred_t = self.out_dense(dec_seq2) 
        
        return y_pred_t, LSTM2LayerTFState(h1=h1_out, c1=c1_out, h2=h2_out, c2=c2_out)

    def train_step(self, data):

        # X_train => X.shape => (batch_size, input_seq_len, feature_dim )
        # Y_train => Y.shape => (batch_size, output_seq_len, feature_dim )
        X, Y = data  

        # the layers are built assuming that number of features.
        # if you change feature_dim at runtime, the weights are not compatible.
        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        prediction_mode_id = tf.convert_to_tensor(self.rt.prediction_mode_id)  # 0 => ALL | 1 => HORIZON
        T_out = tf.convert_to_tensor(self.rt.t_out)
        T_hor = tf.convert_to_tensor(self.rt.horizon)
        
        # IF  PREDICTION_MODE == ALL  (index 0) while condition is t < T_out
        # IF  PREDICTION_MODE == HOR  (index 1) while condition is t < T_hor
        T = tf.cond(
            tf.equal(prediction_mode_id, 0),
            lambda: T_out,
            lambda: T_hor,
        )

        with tf.GradientTape() as tape:
            # return the internal states from encoder 
            state = self._encode(X)
        
            dec_t = self._init_dec0(X)

            # array with T elements and each element have shape(batch_size, feature_dim)
            # batch_size is None because is dinamic 
            ta = tf.TensorArray(
                dtype=Y.dtype,
                size=T,
                element_shape=tf.TensorShape([None, self.feature_dim]),
            )
            
            # ta = shape (batch_size , T , feature_dim)
            # print(ta.element_shape) = shape (batch_size, feature_dim) ,  ta.size() = T_out

            t0 = tf.constant(0, tf.int32)

            def cond(t, dec_t, state, ta):
                return t < T

            def body(t, dec_t, state, ta):
                y_pred_t, state = self._decode_step(dec_t, state)    
                # y_pred_t.shape(batch_size, 1, feature_dim)

                y_pred_t_2d = tf.squeeze(y_pred_t, axis=1)               
                # y_pred_t_2d.shape(batch_size, feature_dim)

                ta = ta.write(t, y_pred_t_2d)
                # ta.shape(index = t , element.shape(batch_size, feature_dim))

                # Ground truth Y.shape(batch_size, timesteps, feature_dim)
                
                y_true_2d = tf.gather(Y, t, axis=1)             
                # y_true_2d.shape(batch_size, feature)
                
                # where index = t in y_true_2d and y_pred_t_2d
                y_true_t = tf.expand_dims(y_true_2d, axis=1)
                # y_true_t.shape(batch_size, 1, feature)

                dec_t = self.apply_strategy_step_wise(y_true_t = y_true_t, y_pred_t = y_pred_t)
                
                return t + 1, dec_t, state, ta

            # the parameter given in input change in the loop 
            # the constant parameter are taken by the father function
            _, _, _, ta = tf.while_loop(cond, body, [t0, dec_t, state, ta], parallel_iterations=1)

            # tf.print(ta.stack())
            # stack: (T, N, D) -> transpose: (N, T, D)
            Y_pred = tf.transpose(ta.stack(), [1, 0, 2])
            Y_pred = tf.slice(Y_pred, [0, 0, 0], [-1, T_hor, -1])

            # Y_true: (N, T_eff, D)
            Y_true = tf.slice(Y, [0, 0, 0], [-1, T_hor, -1])

            Y_pred = tf.cond(
                tf.equal(prediction_mode_id, 0),
                lambda: tf.slice(Y_pred, [0, 0, 0], [-1, T_hor, -1]),
                lambda: Y_pred,
            )
            tf.print(Y_pred)
            tf.print(Y_true)

            loss = self.compute_loss(y=Y_true, y_pred=Y_pred)
                
        # calculate gradients (backward pass)
        grads = tape.gradient(loss, self.trainable_variables) # type: ignore[reportCallIssue]

        # filter the None gradients => take the variables that have gradients
        grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
        if not grads_and_vars:
            raise RuntimeError("There are no gradients")
        
        # gradient clipping => prevents gradient explosions by limiting the global norm
        g_list, v_list = zip(*grads_and_vars)
        g_list, _ = tf.clip_by_global_norm(g_list, self.current_clip_norm)
        
        # backpropagation => update weights
        self.optimizer.apply_gradients(zip(g_list, v_list))

        self.loss_tracker.update_state(loss)      
        # update metrics (includes the metric that tracks the loss)
        self.compute_metrics(x = X, y = Y_true, y_pred = Y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X, Y = data

        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        T_out = tf.cast(self.rt.t_out, tf.int32)

        state = self._encode(X)
        dec_t = self._init_dec0(X)

        ta = tf.TensorArray(
            dtype=Y.dtype, size=T_out,  
            element_shape=tf.TensorShape([None, self.feature_dim]),
        )
        
        t0 = tf.constant(0, dtype=tf.int32)

        def cond(t, dec_t, state, ta):
            return t < T_out

        def body(t, dec_t, state, ta):
            y_t, state = self._decode_step(dec_t, state)
            y_t_nd = tf.squeeze(y_t, axis=1)
            ta = ta.write(t, y_t_nd)

            y_true = tf.gather(Y, t, axis=1)
            y_true_t = tf.expand_dims(y_true, axis=1)

            return t + 1, y_true_t, state, ta

        _, _, _, ta = tf.while_loop(cond, body, [t0, dec_t, state, ta], parallel_iterations=1)


        Y_pred = tf.transpose(ta.stack(), [1, 0, 2])
        Y_true = tf.slice(Y, [0, 0, 0], [-1, T_out, -1])  # stesso slicing del train!

        tf.print(Y_pred)
        tf.print(Y_true)

        val_loss = self.compute_loss(y=Y_true, y_pred=Y_pred)

        self.loss_tracker.update_state(val_loss)

        self.compute_metrics(x=X, y=Y_true, y_pred=Y_pred)

        return {m.name: m.result() for m in self.metrics}
