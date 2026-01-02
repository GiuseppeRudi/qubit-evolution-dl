from ..model.model_config import ModelConfig
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from ..registry import register_model
from typing import cast
from ..model.rnn_config import RNNConfig

@register_model("LSTM", "SEQ2SEQ")
def build_rnn_model(x_train , y_train , model_cfg: ModelConfig):

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

    enc_seq = LSTM(latent_dim, return_sequences=True, name = "enc_lstm_1")(encoder_inputs)
    _, h, c = LSTM(latent_dim, return_state=True , name = "enc_lstm_2")(enc_seq)

    enc_states = [h, c]
    # --- Decoder  (2 stacked LSTM) ---

    # the encoder need the 3d array so that we use this line to repeat the context vector h for each output time step
    # h(batch, latent_dim) => decoder_input(batch, output_seq_len, latent_dim)
    # decoder_input = RepeatVector(output_seq_len)(h)
    dec_seq_1, d_h_1 , d_c_1 = LSTM(latent_dim, return_sequences=True, return_state=True, name = "dec_lstm_1")(decoder_input , initial_state = enc_states)

    # RNN encoder-decoder seq2seq => final states of encoder as initial states of decoder => inizial_state=[h, c]
    dec_seq_2, d_h_2, d_c_2 = LSTM(latent_dim, return_sequences=True, return_state=True, name = "dec_lstm_2")(dec_seq_1, initial_state = enc_states)

    decoder_outputs = Dense(feature_dim, name ="out_dense")(dec_seq_2) 
    
    model = Model([encoder_inputs, decoder_input], decoder_outputs)

    model.compile(optimizer=model_cfg.compile.optimizer, loss=model_cfg.compile.loss, metrics = model_cfg.compile.metrics)

    return model

