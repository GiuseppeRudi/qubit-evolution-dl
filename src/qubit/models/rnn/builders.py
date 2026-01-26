from pathlib import Path
from typing import Optional, cast
from ...enums.prediction_mode import PredictionMode
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed

from ...dataclasses.model_config import ModelConfig
from ...dataclasses.rnn_config import RNNConfig

from ...utils.registry import register_model
from .step_wise_lstm_model import StepWiseLstmModel
from .full_seq_lstm_model import FullSeqLstmModel
from ...core.core import build_optimizer

from ...enums.decoder_mode import DecoderMode
from ...enums.model_type import ModelType
from ...enums.model_variant import ModelVariant


from pathlib import Path
from typing import Optional, cast
import keras

# to convert thre predcition enum into an integer index for graph mode 
prediction_mode_id = {
    PredictionMode.ALL.value: 0,
    PredictionMode.HORIZON.value: 1,
}

@register_model(ModelType.LSTM, ModelVariant.SEQ2SEQ, DecoderMode.FULL_SEQ)
def build_lstm_full_seq_custom_model(
        x_train, # X.shape (n_windows , input_seq_len , feature_dim)
        y_train, # Y.shape (n_windows , output_seq_len  , feature_dim)
        model_cfg: ModelConfig,
        prediction_mode: PredictionMode,
        model_path: Optional[str] # used only when we want to load a pretrained model instead is None
    ) -> FullSeqLstmModel:

    # dimension of the hidden state and cell state (the same)
    latent_dim = cast(RNNConfig, model_cfg.params).latent_dim


    feature_dim = x_train.shape[2]
    # same thing feature_dim = y_train.shape[2]

    # number of time steps to predict 
    t_out = y_train.shape[1]

    # number of time steps to give in input 
    t_in = x_train.shape[1]

    model = FullSeqLstmModel(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        start_mode=model_cfg.inference.start_mode, 
        prediction_mode_id=prediction_mode_id[prediction_mode],
        t_out=t_out,
    )

    optimizer = build_optimizer(
        model_cfg.compile.optimizer,
        model_cfg.compile.learning_rate
    )

    model.compile(
        loss = model_cfg.compile.loss,
        metrics= model_cfg.compile.metrics,
        optimizer=optimizer,
        run_eagerly=model_cfg.compile.run_eagerly,
    )

    model.build(tf.TensorShape([None, t_in, feature_dim]))

    if model_path is not None:
        model.load_weights(Path(model_path) / "model.weights.h5")

    return model


@register_model(ModelType.LSTM, ModelVariant.SEQ2SEQ, DecoderMode.STEP_WISE)
def build_lstm_step_wise_model(
        x_train, # x_train.shape(num_windows, input_seq_len, feature_dim)
        y_train, # y_train.shape(num_windows, output_seq_len, feature_dim)
        model_cfg: ModelConfig,
        prediction_mode: PredictionMode,  
        model_path: Optional[str] = None
    ) -> StepWiseLstmModel:
    
    latent_dim = cast(RNNConfig, model_cfg.params).latent_dim
    feature_dim = int(x_train.shape[2])

    # number of time steps to predict 
    t_out = y_train.shape[1]

    # number of time steps to give in input 
    t_in = x_train.shape[1]
    
    model = StepWiseLstmModel(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        start_mode=model_cfg.inference.start_mode,
        prediction_mode_id=prediction_mode_id[prediction_mode],
        t_out = t_out
    )

    model.build(tf.TensorShape([None, t_in, feature_dim]))

    optimizer = build_optimizer(
        model_cfg.compile.optimizer,
        model_cfg.compile.learning_rate,
    )

    model.compile(
        loss = model_cfg.compile.loss,
        metrics= model_cfg.compile.metrics,
        optimizer=optimizer,
        run_eagerly=model_cfg.compile.run_eagerly
    )

    if model_path is not None:
        model.load_weights(Path(model_path) / "model.weights.h5")

    return model


