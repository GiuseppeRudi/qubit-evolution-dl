from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import keras
from .base import BaseAutoregressiveAdapter, StartMode

from ..models.rnn.lstm2_layer_state import LSTM2LayerTFState

from ..enums.inference_mode import InferenceMode
from ..enums.verbose_mode import VerboseMode

from ..utils.layers_names import ENC_LSTM_1, ENC_LSTM_2, DEC_LSTM_1, DEC_LSTM_2, OUT_DENSE

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
    out_dense = trained_model.get_layer(OUT_DENSE)

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
    def __init__(self, trained_model: keras.Model, *, out_steps: int, start_mode: StartMode, inference_mode: InferenceMode, verbose: VerboseMode):
        super().__init__(out_steps=out_steps, start_mode=start_mode, inference_mode=inference_mode)
        self.verbose = verbose

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