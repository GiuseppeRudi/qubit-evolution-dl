from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import keras
from .base import AutoregressiveAdapter, StartMode


@dataclass
class LSTM2LayerState:
    """State for 2-layer decoder LSTM: (h1,c1,h2,c2), each shape (N, latent_dim)."""
    h1: np.ndarray
    c1: np.ndarray
    h2: np.ndarray
    c2: np.ndarray


def build_inference_models(
    trained_model: keras.Model,
    *,
    enc_lstm_1_name: str = "enc_lstm_1",
    enc_lstm_2_name: str = "enc_lstm_2",
    dec_lstm_1_name: str = "dec_lstm_1",
    dec_lstm_2_name: str = "dec_lstm_2",
    out_dense_name: str = "out_dense",
) -> tuple[keras.Model, keras.Model, int, int]:
    """
    Build encoder_model and decoder_model for autoregressive inference,
    reusing trained layers/weights.

    Returns: (encoder_model, decoder_model, feature_dim, latent_dim)
    """

    # trained_model.input[0].shape : (None, T_in, D)
    # encoder input: (T_in, D)
    enc_input_shape = trained_model.inputs[0].shape[1:]


    feature_dim = int(enc_input_shape[-1])

    # Functinal API to build inference models reusing trained layers
    enc_lstm_1 = trained_model.get_layer(enc_lstm_1_name)
    enc_lstm_2 = trained_model.get_layer(enc_lstm_2_name)
    dec_lstm_1 = trained_model.get_layer(dec_lstm_1_name)
    dec_lstm_2 = trained_model.get_layer(dec_lstm_2_name)
    out_dense = trained_model.get_layer(out_dense_name)

    latent_dim = dec_lstm_1.units

    # IMPORTANT: dec_lstm_2 must return_state for step-by-step decoding
    if not dec_lstm_2.return_state:
        raise ValueError("dec_lstm_2 must be created with return_state=True for free-running inference" )

    # --- Encoder inference: X -> (h, c)

    # minimal model needed to get encoder final states used for decoder init
    enc_in = keras.Input(shape=enc_input_shape, name="enc_in_inf")
    x , h1, c1= enc_lstm_1(enc_in)
    _, h2, c2 = enc_lstm_2(x)

    # create a model that take in input the encoder input and return the hidden and cell states
    encoder_model = keras.Model(enc_in, [h1, c1,h2,c2], name="encoder_inf")

    # --- Decoder inference (one step)

    # the differnce from the training decoder is here 
    # we use the different input shape for one time step decoding not for all timesteps at once
    dec_in_t = keras.Input(shape=(1, feature_dim), name="dec_in_t")
    
    h1_in = keras.Input(shape=(latent_dim,), name="h1_in")
    c1_in = keras.Input(shape=(latent_dim,), name="c1_in")
    h2_in = keras.Input(shape=(latent_dim,), name="h2_in")
    c2_in = keras.Input(shape=(latent_dim,), name="c2_in")

    dec_seq1, h1_out, c1_out = dec_lstm_1(dec_in_t, initial_state=[h1_in, c1_in])
    dec_seq2, h2_out, c2_out = dec_lstm_2(dec_seq1, initial_state=[h2_in, c2_in])

    # final dense layer to get the prediction
    y_out = out_dense(dec_seq2)  # (N,1,D)

    decoder_model = keras.Model(
        [dec_in_t, h1_in, c1_in, h2_in, c2_in],
        [y_out, h1_out, c1_out, h2_out, c2_out],
        name="decoder_inf",
    )

    return encoder_model, decoder_model, feature_dim, latent_dim


class FullSeqLstmAdapter(AutoregressiveAdapter):

    def __init__(self, trained_model: keras.Model, *, verbose , layer_names: dict | None = None ):
        layer_names = layer_names or {}
        self.verbose = verbose
        (self.encoder_model,
         self.decoder_model,
         self._feature_dim,
         self._latent_dim) = build_inference_models(trained_model, **layer_names)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode(self, X: np.ndarray, *, batch_size: int) -> LSTM2LayerState:
        # X = ensure_3d(X).astype(np.float32, copy=False)
        # h1,c1, h2 , c2 from encoder
        h1, c1, h2,c2 = self.encoder_model.predict(X, batch_size=batch_size, verbose= self.verbose)
        return LSTM2LayerState(h1=h1, c1=c1, h2=h2, c2=c2)

    def init_decoder_input(self, X: np.ndarray, *, start_mode: StartMode) -> np.ndarray:
        # X = ensure_3d(X).astype(np.float32, copy=False)

        N, _, D = X.shape
        if start_mode == StartMode.LAST_X:
            return X[:, -1:, :]  # (N,1,D)
        # default zeros
        return np.zeros((N, 1, D), dtype=np.float32)

    def step(
        self,
        dec_t: np.ndarray,
        state: LSTM2LayerState,
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, LSTM2LayerState]:
        dec_t = np.asarray(dec_t, dtype=np.float32)
        y_t, h1, c1, h2, c2 = self.decoder_model.predict(
            [dec_t, state.h1, state.c1, state.h2, state.c2],
            batch_size=batch_size,verbose= self.verbose)
        
        new_state = LSTM2LayerState(h1=h1, c1=c1, h2=h2, c2=c2)
        return y_t, new_state


