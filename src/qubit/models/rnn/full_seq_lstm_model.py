from __future__ import annotations

import tensorflow as tf
import keras
from keras import layers

from ...enums.start_mode import StartMode
from ..strategy_chooser import StrategyChooserModel
from .lstm2_layer_state import LSTM2LayerTFState

from ...utils.layers_names import ENC_LSTM_1, ENC_LSTM_2, DEC_LSTM_1, DEC_LSTM_2, OUT_DENSE

class FullSeqLstmModel(StrategyChooserModel):

    def __init__(self, *, feature_dim: int, latent_dim: int, start_mode: StartMode, t_out: int, prediction_mode_id : int):
        super().__init__(t_out = t_out, prediction_mode_id = prediction_mode_id)
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim # dimension of hidden states and cell states 
        self.start_mode = start_mode # inizialize the encoder with last_x or zeros 

        # Inizialize the layers of Model 

        # encoder (2 stacked LSTM)
        self.enc_lstm_1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name=ENC_LSTM_1)
        self.enc_lstm_2 = layers.LSTM(latent_dim, return_state=True, name=ENC_LSTM_2)

        # decoder full-seq (input: (B, T_out, D))
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
        # Encoder build

        # input_shape: (batch_size, input_seq_len, feature_dim)
        # input shape as parameter is needed to respect the build in function

        # we insert only the last dimension because only this is important for the weights
        # enc_lstm1 return sequence of dimension (batch_size, t_in , latent_dim)
        self.enc_lstm_1.build((None, None, self.feature_dim))  # (batch_size , t_in, feature_dim)

        # since enc_lstm_2 takes in input the sequence of enc_lstm1 so the 3rd dimension is latent_dim
        self.enc_lstm_2.build((None, None, self.latent_dim))  # (batch, t_in , latent_dim)

        # Decoder build
        # decoder take in input the dec_in (previous predictions with possibility to apply a different strategies)
        # at the start dec_in depends on the start_mode (zeros or last x)
        # dec_in.shape(batch_size,t_out, feauture_dim)
        self.dec_lstm_1.build((None, None, self.feature_dim))       

        # dec_lstm_2 take the dec_lstm_1 output sequence of dimension (batch_size, t_out, latent_dim)
        self.dec_lstm_2.build((None, None, self.latent_dim))   

        # dec_lstm_2 output sequence of dimension (batch_size, t_out, latent_dim)
        self.out_dense.build((None, None, self.latent_dim))     # (batch_size, t_out, feature_dim)

        super().build(input_shape)

    # called for each batch
    def call(self, inputs):

        X, dec_in = inputs

        # X.shape(batch_size, input_seq_len , feature_dim)
        # dec_in(batch_size,  t = T_out || T_hor, feature_dim ) 
        states = self._encode(X)
        # state are the hidden states and cell states of the two encoder layers 

        # return y_pred from the decoder with shape (batch_size, t = T_out || T_hor , feature_dim)
        return self._decode_full_seq(dec_in, states)

    def _encode(self, X: tf.Tensor) -> LSTM2LayerTFState:
        
        # X.shape(batch_size, input_seq_len , feature_dim)

        # enc_lstm_1 return_sequence = True , return_state = True
        x_seq, h1, c1 = self.enc_lstm_1(X)

        # x_seq.shape(batch_size, input_seq_len, latent_dim)
        # h1 and c1 .shape(batch_size, latent_dim)

        # enc_lstm_2 return_sequence = False , return_state = True 
        # take in input the output sequence of enc_lstm_1
        _, h2, c2 = self.enc_lstm_2(x_seq)
        # h2 and c2 .shape(batch_size, latent_dim)

        return LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)

    def _decode_full_seq(self, dec_in: tf.Tensor, states: LSTM2LayerTFState) -> tf.Tensor:
        # dec_in(batch_size,  t = T_out || T_hor, feature_dim )  => 
        # ! Decoder input sequence (e.g. teacher forcing: shifted y_true; masked modeling: masked y_true)

        # ! return_state = True needed only for inference adapter for autoregressive predictions
        # dec_lstm1 => return_state = True and return_sequence = True
        # dec_lstm2 => return_state = True and return_sequence = True

        # initialization of hidden state and cell state equal to the hidden state and cell state from encoder 1 layer
        dec_seq_1, _, _ = self.dec_lstm_1(dec_in, initial_state=[states.h1, states.c1])

        # dec_seq_1.shape(batch_size, t = T_out || T_hor, latent_dim)
        
        # initialization of hidden state and cell state equal to the hidden state and cell state from encoder 2 layer
        dec_seq_2, _, _ = self.dec_lstm_2(dec_seq_1, initial_state=[states.h2, states.c2])

        # dec_seq_2.shape(batch_size, t = T_out || T_hor, latent_dim)

        # needed to change the third dimensions 
        # latent_dim => feature_dim
        y_pred = self.out_dense(dec_seq_2)  
        
        # y_pred.shape(batch_size, t = T_out || T_hor, feature_dim)
        return y_pred

    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:
        
        # INFO => Decoder takes in input (batch_size , t_out , feature_dim) in the apply strategy function, not here

        # Inizialize decoder input at the t (timestep) = 0 
        # X.shape(batch_size , input_seq_len, feature_dim)

        #  ! Initialization of subsequent timesteps occurs when entering strategy functions.

        if self.start_mode == StartMode.LAST_X:
            
            input_seq_len = tf.shape(X)[1]

            # take the last timestep 
            last = tf.gather(X, input_seq_len - 1, axis=1) 
            # last.shape(batch_size, feature_dim)

            # add the timestep dimension 
            return tf.expand_dims(last, axis=1)  # (Batch_size, 1 , feature_dim)
        
        batch_size = tf.shape(X)[0]
        feature_dim = self.feature_dim
        
        # returns an array with shape(batch_size, 1 , feature_dim) all filled with zeros
        return tf.zeros(tf.stack([batch_size, 1, feature_dim]), dtype=X.dtype)

    # each batch call this function
    def train_step(self, data):

        # X_train => X.shape => (batch_size, input_seq_len, feature_dim )
        # Y_train => Y.shape => (batch_size, output_seq_len, feature_dim )
        X, Y = data

        # the layers are built assuming that number of features.
        # if you change feature_dim at runtime, the weights are not compatible.
        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        # tf.print(type(self.rt.t_out)) => keras.backend.Variable => wrapper of tf.Variable
        # we prefer to work with tf.Tensor 

        # we have two options 
        # use tf.cast to convert the tf.Variable into tf.Tensor automatically and ensure the dtype 
        # T_out = tf.cast(self.rt.t_out, tf.int32)
        # use tf.convert_to_tensor and optional use the parameter dtype
        # T_out = tf.convert_to_tensor(self.rt.t_out, dtype=tf.int32)
        
        # self.rt.horizon when is = -1 so self.rt.horizon == self.rt.t_out
        T_hor = tf.convert_to_tensor(self.rt.horizon)

        # inititialize to zeros or last_x based on the start_mode parameter
        dec0 = self._init_dec0(X)         

        prediction_mode_id = tf.convert_to_tensor(self.rt.prediction_mode_id)  # 0 => ALL | 1 => HORIZON
        
        # dec_in(batch_size,  t = T_out || T_hor, feature_dim )  
        
        # ! in full_seq  dec_in is the prediction help that contains all the information
        # ! that the decoder must predict using ground truth (with the possibility of masking) 

        def run_all():
            
            # decoder input  dec_in.shape(batch_size , T_out == output_seq_len, feature_dim)
            dec_in = self.apply_strategy_full_seq(Y, dec0=dec0)

            # call the call function
            Y_pred = self([X, dec_in])                      
            # Y_pred.shape(batch_size, T_out == output_seq_len , feature_dim)

            # so we predict to all T_out to stabilize the graph and not occur the tracing
            # but we return only the part we need (horizon)
            return Y[:, :T_hor, :], Y_pred[:, :T_hor, :]

        def run_hor_only():
            # Y.shape(batch_size, T_out, feature_dim)
            # Y_hor(batch_size , T_hor, feature_dim)
            Y_hor = Y[:, :T_hor, :]

            dec_in = self.apply_strategy_full_seq(Y_hor, dec0=dec0) 
            # decoder_input dec_in.shape(batch_size, T_hor, feature_dim)  

            # call the call function 
            Y_pred = self([X, dec_in]) 
                                
            return Y_hor, Y_pred

        # forward pass => compute predictions and loss => register gradients
        # It constructs a computational graph that tracks how the loss depends on the weights
        # This then allows the gradients to be calculated
        with tf.GradientTape() as tape:
            Y_true, Y_pred = tf.cond(tf.equal(prediction_mode_id, 0), run_all, run_hor_only)

            # Y_true.shape(batch_size , T_hor , feature_dim)
            # Y_pred.shape(batch_size , T_hor , feature_dim)

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
        # at the end of each epoch we evaluate the metrics with the validation splits
        
        X, Y = data
        
        # X_val => X.shape => (batch_size, input_seq_len, feature_dim )
        # Y_val => Y.shape => (batch_size, output_seq_len, feature_dim )

        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        # ! dec_in is always the ground truth => teacher forcing + dec0
        # ! Y_pred second dimension is always => horizon => coerent with train_step

        # self.rt.horizon when is = -1 so self.rt.horizon == self.rt.t_out
        T_hor = tf.convert_to_tensor(self.rt.horizon)

        # inititialize to zeros or last_x based on the start_mode parameter
        dec0 = self._init_dec0(X)         

        prediction_mode_id = tf.convert_to_tensor(self.rt.prediction_mode_id)  # 0 => ALL | 1 => HORIZON
        
        # dec_in(batch_size,  t = T_out || T_hor, feature_dim )  
        
        # ! in full_seq  dec_in is the prediction help that contains all the information
        # ! that the decoder must predict using ground truth (with the possibility of masking) 

        def run_all():
            
            # decoder input  dec_in.shape(batch_size , T_out == output_seq_len, feature_dim)
            dec_in = self.apply_strategy_full_seq(Y, dec0=dec0)

            # call the call function
            Y_pred = self([X, dec_in])                      
            # Y_pred.shape(batch_size, T_out == output_seq_len , feature_dim)

            # so we predict to all T_out to stabilize the graph and not occur the tracing
            # but we return only the part we need (horizon)
            return Y[:, :T_hor, :], Y_pred[:, :T_hor, :]

        def run_hor_only():
            # Y.shape(batch_size, T_out, feature_dim)
            # Y_hor(batch_size , T_hor, feature_dim)
            Y_hor = Y[:, :T_hor, :]

            dec_in = self.apply_strategy_full_seq(Y_hor, dec0=dec0) 
            # decoder_input dec_in.shape(batch_size, T_hor, feature_dim)  

            # call the call function 
            Y_pred = self([X, dec_in]) 
                                
            return Y_hor, Y_pred

        self.rt.phase_id.assign(0) # teacher forcing strategy

        Y_true, Y_pred = tf.cond(tf.equal(prediction_mode_id, 0), run_all, run_hor_only)
        
        val_loss = self.compute_loss(y=Y_true, y_pred=Y_pred)
        self.loss_tracker.update_state(val_loss)

        self.compute_metrics(x=X, y=Y_true, y_pred=Y_pred)

        return {m.name: m.result() for m in self.metrics}
