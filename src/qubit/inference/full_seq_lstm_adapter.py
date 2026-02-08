from __future__ import annotations

import tensorflow as tf
import keras
from .base_adapter import BaseAutoregressiveAdapter

from ..models.rnn.lstm2_layer_state import LSTM2LayerTFState

from ..enums.inference_mode import InferenceMode
from ..enums.verbose_mode import VerboseMode

from ..utils.layers_names import ENC_LSTM_1, ENC_LSTM_2, DEC_LSTM_1, DEC_LSTM_2, OUT_HEAD

# return functional API model for inference because we don't need very complex logic here instead of subclassing full_seq trained_model
def build_inference_models(trained_model: keras.Model) -> tuple[keras.Model, keras.Model, int, int]:
    """
    Build encoder_model and decoder_model for autoregressive inference,
    reusing trained layers/weights.

    Returns: (encoder_model, decoder_model, feature_dim, latent_dim)
    """

    # trained_model.input[0].shape : (batch_size , t_in, feature_dim) => encoder inputs 
    # trained_model.input[1].shape : (batch_size , t_out, feature_dim) => decoder inputs 
        
    feature_dim = trained_model.feature_dim
    latent_dim = trained_model.latent_dim

    # take the trained layers 
    enc_lstm_1 = trained_model.get_layer(ENC_LSTM_1)
    enc_lstm_2 = trained_model.get_layer(ENC_LSTM_2)
    dec_lstm_1 = trained_model.get_layer(DEC_LSTM_1)
    dec_lstm_2 = trained_model.get_layer(DEC_LSTM_2)
    out_dense = trained_model.get_layer(OUT_HEAD)

    # ! dec_lstm_2 must return_state for step-by-step decoding
    if not dec_lstm_2.return_state:
        raise ValueError("dec_lstm_2 must be created with return_state=True for free-running inference" )

    # --- Encoder inference: X -> (h, c)
    
    # !  keras.Input automatically insert batch_size as third dimension in the first position 
    enc_in = keras.Input(shape=(None, feature_dim), name="enc_in_inf")
    # enc_in.shape(batch_size , t_in, feature_dim)

    # enc_lstm_1 return_state = true and return_sequence = true 
    # x.shape(batch_size, t_in, latent_dim)
    # h1 and c1.shape(batch_size , latent_dim)
    x, h1, c1= enc_lstm_1(enc_in)

    # enc_lstm_2 return_state = True and return sequence = True
    # h2 and c2 .shape(batch_size, latent_dim)
    _, h2, c2 = enc_lstm_2(x)

    # create a model that take in input the encoder input 
    # and return the hidden and cell states to send them to decoder_model
    encoder_model = keras.Model(enc_in, [h1, c1, h2, c2], name="encoder_inf")

    # --- Decoder inference (one step)

    # in the trained_mode FULL SEQ the decoder input take this shape
    # dec_in.shape(batch_size, t_out, feauture_dim) 

    # the differnce from the training decoder FULL_SEQ  is here 
    # we use the different input shape for one time step decoding not for all timesteps at once

    dec_in_t = keras.Input(shape=(1, feature_dim), name="dec_in_t")
    # tf.print(dec_in_t.shape)   # (batch_size, 1 ,feature_dim)
    
    
    h1_in = keras.Input(shape=(latent_dim,), name="h1_in")
    c1_in = keras.Input(shape=(latent_dim,), name="c1_in")
    h2_in = keras.Input(shape=(latent_dim,), name="h2_in")
    c2_in = keras.Input(shape=(latent_dim,), name="c2_in")
    # h1_in , c1_in , h2_in , c2_in .shape(batch_size, latent_dim)

    dec_seq1, h1_out, c1_out = dec_lstm_1(dec_in_t, initial_state=[h1_in, c1_in])
    # dec_seq1.shape(batch_size, 1, latent_dim)

    dec_seq2, h2_out, c2_out = dec_lstm_2(dec_seq1, initial_state=[h2_in, c2_in])
    # dec_seq2.shape(batch_size, 1, latent_dim)

    # h1_out, c1_out, h2_out, c2_out .shape(batch_size, latent_dim)

    y_out = out_dense(dec_seq2)  
    # y_out.shape(batch_size, 1, feature_dim)

    # decoder_mode takes in input the dec_in_t where at the timestep t = 0 is (zeros or last_x based on the startMode)
    # and the hidden states and cell states from the encoder input 
    # in the next timesteps dec_in_t and the hidden states and cell states cames from the decoder output at the previous timesteps t-1

    # ? the input at the timestep t => is the output from the timestep t-1
    # ? the output at timestep t => will be the input at the timestep t+1
    
    decoder_model = keras.Model(
        [dec_in_t, h1_in, c1_in, h2_in, c2_in],
        [y_out, h1_out, c1_out, h2_out, c2_out],
        name="decoder_inf",
    )
    return encoder_model, decoder_model, feature_dim, latent_dim

class FullSeqLstmAdapter(BaseAutoregressiveAdapter):
    def __init__(self, trained_model: keras.Model, *, out_steps: int, inference_mode: InferenceMode):
        super().__init__(out_steps=out_steps, inference_mode=inference_mode,  name="full_seq_lstm_adapter")
        self.trained_model = trained_model

        (self.encoder_model,
         self.decoder_model,
         self._feature_dim,
         self._latent_dim) = build_inference_models(trained_model)


    def encode(self, X: tf.Tensor) -> LSTM2LayerTFState:
        # X.shape(batch_size, input_seq_len, feature_dim)

        h1, c1, h2, c2 = self.encoder_model(X, training=False)
        # h1 and c1 from encoder layer 1 
        # h2 and c2 from encoder layer 2 

        # h* and c* shape(batch_size, latent_dim)
        return LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)

    def step(self, dec_t: tf.Tensor, state: LSTM2LayerTFState):
        
        # dec_t.shape(batch_size, 1, feature_dim)
            # if t != 0 dec_t => previous prediction, instead the start mode

        # if t = 0 hidden states and cell states from the encoder
        # if t > 0 hidden states and cell states from the decoder at timestep t-1

        # state => LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)
        # h* and c* are shape(batch_size, latent_dim)
        
        y_t, h1, c1, h2, c2 = self.decoder_model(
            [dec_t, state.h1, state.c1, state.h2, state.c2],
            training=False,
        )
        
        # y_t.shape(batch_size, 1 , feature_dim)
        return y_t, LSTM2LayerTFState(h1=h1, c1=c1, h2=h2, c2=c2)
    
    
    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:
        return self.trained_model._init_dec0(X)
    

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        
        # self.outsteps 
        # if prediction_mode == ALL outsteps = output_seq_len
        # if prediction_mode == HORIZON outsteps = horizon for a specific phase (strategy)

        # X.shape(batch_size, input_seq_len, feature_dim)
  
        # if inference_mode == TEACHER_FORCING , inputs = (X, y_true)
        if isinstance(inputs, (tuple, list)):
            # Y_true (batch_size, t, feature_dim)
            # if prediction_mode == ALL t = output_seq_len
            # if prediction_mode == HORIZON t = horizon for a specific phase (strategy)
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
            
            y_pred_t = tf.expand_dims(y_pred_t_2d, axis=1)
            # y_pred_t.shape(batch_size, 1,feature_dim)

            # dec_t will be the decoder input for the next iteration (t+1) 
            
            if use_tf:
                dec_t = tf.slice(Y_true, [0, t, 0], [-1, 1, -1])
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