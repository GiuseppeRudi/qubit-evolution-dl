from pathlib import Path
from ..model.model_config import ModelConfig
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from ..registry import register_model
from typing import Optional, cast
from ..model.rnn_config import RNNConfig

from .step_wise_lstm_model import StepWiseLstmModel

from ..enums.decoder_mode import DecoderMode
from ..enums.model_type import ModelType
from ..enums.model_variant import ModelVariant

from ..core.core import build_optimizer

@register_model(ModelType.LSTM, ModelVariant.SEQ2SEQ, DecoderMode.FULL_SEQ)
def build_lstm_full_seq_model(x_train , y_train , model_cfg: ModelConfig, model_path: Optional[str] = None) -> Model:

    latent_dim = cast(RNNConfig,model_cfg.params).latent_dim

    # x_train.shape == (N, input_seq_len, feature_dim)
    # y_train.shape == (N, output_seq_len, feature_dim)

    input_seq_len , feature_dim = x_train.shape[1], x_train.shape[2]
    output_seq_len  = y_train.shape[1]

    input_shape = (input_seq_len, feature_dim)
    output_shape = (output_seq_len, feature_dim)

    # --- Encoder (2 stacked LSTM) ---

    # analyze one trajectory at time
    encoder_inputs = Input(shape=input_shape)
    decoder_input = Input(shape = output_shape)

    # return_sequences = True => return the output for all timesteps => output(batch, timesteps, latent_dim)
    # return_sequences = False (default) => return only the last output => output(batch, latent_dim)

    # return_state = False (default) => don't return the hidden states 
    # return_state = True => return the hidden state and cell state => state_h(batch, latent_dim), state_c(batch, latent_dim)

    enc_seq, h1, c1 = LSTM(latent_dim, return_sequences=True, return_state = True, name = "enc_lstm_1")(encoder_inputs)
    _, h2, c2 = LSTM(latent_dim, return_state=True , name = "enc_lstm_2")(enc_seq)

    # --- Decoder  (2 stacked LSTM) ---

    # the encoder need the 3d array so that we use this line to repeat the context vector h for each output time step
    # h(batch, latent_dim) => decoder_input(batch, output_seq_len, latent_dim)
    # decoder_input = RepeatVector(output_seq_len)(h)
    dec_seq_1, d_h_1, d_c_1 = LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm_1")(
        decoder_input, initial_state=[h1, c1]
    )

    dec_seq_2, d_h_2, d_c_2 = LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm_2")(
        dec_seq_1, initial_state=[h2, c2]
    )

    decoder_outputs = Dense(feature_dim, name ="out_dense")(dec_seq_2) 
    
    model = Model([encoder_inputs, decoder_input], decoder_outputs)

    optimizer = build_optimizer(
        model_cfg.compile.optimizer,
        model_cfg.compile.learning_rate,
        model_cfg.compile.clip_norm
    )

    model.compile(optimizer=optimizer, loss=model_cfg.compile.loss, metrics = model_cfg.compile.metrics)

    if model_path is not None:
        model.load_weights(Path(model_path) / "model.weights.h5")

    return model


@register_model(ModelType.LSTM, ModelVariant.SEQ2SEQ, DecoderMode.STEP_WISE)
def build_lstm_step_wise_model(x_train, y_train, model_cfg: ModelConfig, model_path: Optional[str] = None) -> StepWiseLstmModel:
    latent_dim = cast(RNNConfig, model_cfg.params).latent_dim
    feature_dim = int(x_train.shape[2])

    model = StepWiseLstmModel(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        start_mode=model_cfg.inference.start_mode,
    )

    model.build((None, None, feature_dim))

    optimizer = build_optimizer(
        model_cfg.compile.optimizer,
        model_cfg.compile.learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=model_cfg.compile.loss,
        metrics=model_cfg.compile.metrics,
        run_eagerly=model_cfg.compile.run_eagerly
    )

    if model_path is not None:
        model.load_weights(Path(model_path) / "model.weights.h5")

    return model


