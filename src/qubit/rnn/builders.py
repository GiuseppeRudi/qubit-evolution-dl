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
    output_shape = (output_seq_len, feature_dim)

    # --- Encoder ---

    # analyze one trajectory at time
    encoder_inputs = Input(shape=input_shape)

    # return_sequences = True => return the output for all timesteps => output(batch, timesteps, latent_dim)
    # return_sequences = False (default) => return only the last output => output(batch, latent_dim)

    # return_state = False (default) => don't return the hidden states 
    # return_state = True => return the hidden state and cell state => state_h(batch, latent_dim), state_c(batch, latent_dim)

    _, h, c = LSTM(latent_dim, return_state=True)(encoder_inputs)

    # --- Decoder ---

    # the encoder need the 3d array so that we use this line to repeat the context vector h for each output time step
    # h(batch, latent_dim) => decoder_input(batch, output_seq_len, latent_dim)
    # decoder_input = RepeatVector(output_seq_len)(h)

    

    #TODO we need to use a different approach for example the teacher forcing or autoregressive
    decoder_input = Input(shape = output_shape)

    # RNN encoder-decoder seq2seq => final states of encoder as initial states of decoder => inizial_state=[h, c]
    decoder_seq = LSTM(latent_dim, return_sequences=True)(decoder_input, initial_state=[h, c])

    decoder_outputs = Dense(feature_dim)(decoder_seq) 
    
    model = Model([encoder_inputs, decoder_input], decoder_outputs)

    model.compile(optimizer=model_cfg.compile.optimizer, loss=model_cfg.compile.loss, metrics = model_cfg.compile.metrics)

    return model