from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import keras
from .base import AutoregressiveAdapter, StartMode, ensure_3d


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
    # Dimensions from encoder input: (T_in, D)
    enc_input_shape = trained_model.inputs[0].shape[1:]
    if enc_input_shape is None or len(enc_input_shape) != 2:
        raise ValueError(f"Unexpected encoder input shape: {trained_model.inputs[0].shape}")

    feature_dim = int(enc_input_shape[-1])

    # Fetch layers
    enc_lstm_1 = trained_model.get_layer(enc_lstm_1_name)
    enc_lstm_2 = trained_model.get_layer(enc_lstm_2_name)
    dec_lstm_1 = trained_model.get_layer(dec_lstm_1_name)
    dec_lstm_2 = trained_model.get_layer(dec_lstm_2_name)

    try:
        out_dense = trained_model.get_layer(out_dense_name)
    except ValueError:
        # fallback: last layer
        out_dense = trained_model.layers[-1]

    latent_dim = int(getattr(dec_lstm_1, "units", None) or 0)
    if latent_dim <= 0:
        raise ValueError("Could not infer latent_dim from dec_lstm_1.units")

    # IMPORTANT: dec_lstm_2 must return_state for step-by-step decoding
    if not getattr(dec_lstm_2, "return_state", False):
        raise ValueError(
            "dec_lstm_2 must be created with return_state=True for free-running inference. "
            "In your builder: LSTM(..., return_sequences=True, return_state=True, name='dec_lstm_2')."
        )

    # --- Encoder inference: X -> (h, c)
    enc_in = keras.Input(shape=enc_input_shape, name="enc_in_inf")
    x = enc_lstm_1(enc_in)
    _, h, c = enc_lstm_2(x)
    encoder_model = keras.Model(enc_in, [h, c], name="encoder_inf")

    # --- Decoder inference (one step)
    dec_in_t = keras.Input(shape=(1, feature_dim), name="dec_in_t")
    h1_in = keras.Input(shape=(latent_dim,), name="h1_in")
    c1_in = keras.Input(shape=(latent_dim,), name="c1_in")
    h2_in = keras.Input(shape=(latent_dim,), name="h2_in")
    c2_in = keras.Input(shape=(latent_dim,), name="c2_in")

    dec_seq1, h1_out, c1_out = dec_lstm_1(dec_in_t, initial_state=[h1_in, c1_in])
    dec_seq2, h2_out, c2_out = dec_lstm_2(dec_seq1, initial_state=[h2_in, c2_in])

    y_out = out_dense(dec_seq2)  # (N,1,D)

    decoder_model = keras.Model(
        [dec_in_t, h1_in, c1_in, h2_in, c2_in],
        [y_out, h1_out, c1_out, h2_out, c2_out],
        name="decoder_inf",
    )

    return encoder_model, decoder_model, feature_dim, latent_dim


class Seq2SeqLSTM2LayerAdapter(AutoregressiveAdapter):
    """
    Adapter per il tuo modello:
    - encoder: enc_lstm_1 -> enc_lstm_2 (stati h,c)
    - decoder: dec_lstm_1 (stateful) -> dec_lstm_2 (stateful) -> out_dense
    """

    def __init__(self, trained_model: keras.Model, *, layer_names: dict | None = None):
        layer_names = layer_names or {}
        (self.encoder_model,
         self.decoder_model,
         self._feature_dim,
         self._latent_dim) = build_inference_models(trained_model, **layer_names)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode(self, X: np.ndarray, *, batch_size: int) -> LSTM2LayerState:
        X = ensure_3d(X).astype(np.float32, copy=False)
        # h1,c1 from encoder
        h1, c1 = self.encoder_model.predict(X, batch_size=batch_size)

        # h2,c2 initial zeros (because in training your dec_lstm_2 starts from zeros)
        N = X.shape[0]
        h2 = np.zeros((N, self._latent_dim), dtype=np.float32)
        c2 = np.zeros((N, self._latent_dim), dtype=np.float32)
        return LSTM2LayerState(h1=h1, c1=c1, h2=h2, c2=c2)

    def init_decoder_input(self, X: np.ndarray, *, start_mode: StartMode) -> np.ndarray:
        X = ensure_3d(X).astype(np.float32, copy=False)
        N, _, D = X.shape
        if start_mode == "last_x":
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
            batch_size=batch_size)
        
        new_state = LSTM2LayerState(h1=h1, c1=c1, h2=h2, c2=c2)
        return y_t, new_state

# TODO check predictions are correct whem we plot them