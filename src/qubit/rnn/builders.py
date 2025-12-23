from ..model.model_config import ModelConfig
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from ..registry import register_model
from typing import cast
from ..model.rnn_config import RNNConfig

@register_model("RNN", "SEQ2SEQ")
def build_rnn_model(x_train , y_train , model_cfg: ModelConfig):

    latent_dim = cast(RNNConfig,model_cfg.params).latent_dim

    # x_train.shape == (N, input_seq_len, feature_dim)
    # y_train.shape == (N, output_seq_len, feature_dim)

    input_seq_len , feature_dim = x_train.shape[1], x_train.shape[2]
    output_seq_len  = y_train.shape[1]

    input_shape = (input_seq_len, feature_dim)

    # --- Encoder ---

    # analyze one trajectory at time
    encoder_inputs = Input(shape=input_shape)


    _, h, c = LSTM(latent_dim, return_state=True)(encoder_inputs)

    decoder_input = RepeatVector(output_seq_len)(h)
    decoder_seq = LSTM(latent_dim, return_sequences=True)(decoder_input, initial_state=[h, c])

    decoder_outputs = Dense(feature_dim)(decoder_seq)  # niente TimeDistributed
    model = Model(encoder_inputs, decoder_outputs)

    model = Model(encoder_inputs, decoder_outputs, name=model_cfg.name)
    model.compile(optimizer=model_cfg.compile.optimizer, loss=model_cfg.compile.loss)

    return model