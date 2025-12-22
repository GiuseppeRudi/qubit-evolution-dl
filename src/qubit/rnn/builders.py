from ..model.model_config import ModelConfig
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from ..registry import register_model
from typing import cast
from ..model.rnn_config import RNNConfig

@register_model("RNN", "SEQ2SEQ")
def build_rnn_model(x_train , y_train , model_cfg: ModelConfig):

    latent_dim=cast(RNNConfig,model_cfg.params).latent_dim

    # x_train.shape == (N, input_seq_len, feature_dim)
    # y_train.shape == (N, output_seq_len, feature_dim)

    if x_train.ndim != 3 or y_train.ndim != 3 : 
        raise ValueError(f"Expected 3D tensors. Got x:{x_train.shape}, y:{y_train.shape}")

    input_seq_len , x_feat = x_train.shape[1], x_train.shape[2]
    output_seq_len , y_feat = y_train.shape[1], y_train.shape[2]

    if x_feat != y_feat:
        raise ValueError(f"Feature dim mismatch: x has {x_feat}, y has {y_feat}")

    input_shape = (input_seq_len, x_feat)
    feature_dim = x_feat

    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape)
    _, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_inputs)
    encoder_states = [state_h, state_c]

    # --- Decoder ---
    decoder_input = RepeatVector(output_seq_len)(state_h)
    decoder_lstm = LSTM(latent_dim, return_sequences=True)(decoder_input, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(feature_dim))(decoder_lstm)

    model = Model(encoder_inputs, decoder_outputs, name=model_cfg.name)
    model.compile(optimizer=model_cfg.compile.optimizer, loss=model_cfg.compile.loss)

    return model