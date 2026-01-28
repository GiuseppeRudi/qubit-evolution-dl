from pathlib import Path
from typing import Optional, cast

import tensorflow as tf

from .step_wise_model import StepWiseTrnModel

from ...dataclasses.transformer_config import TransformerConfig
from ...dataclasses.model_config import ModelConfig
from ...utils.registry import register_model

from ...enums.model_type import ModelType
from ...enums.model_variant import ModelVariant
from ...enums.decoder_mode import DecoderMode
from ...enums.prediction_mode import PredictionMode

from ...core.core import build_optimizer

# to convert thre predictions enum into an integer index for graph mode 
prediction_mode_id = {
    PredictionMode.ALL.value: 0,
    PredictionMode.HORIZON.value: 1,
}

@register_model(ModelType.TRN, ModelVariant.SEQ2SEQ, DecoderMode.STEP_WISE)
def build_transformer_step_wise_model(
    x_train,# X.shape (n_windows , input_seq_len , feature_dim)
    y_train, # Y.shape (n_windows , output_seq_len  , feature_dim)
    model_cfg: ModelConfig,
    prediction_mode: PredictionMode,
    model_path: Optional[str] = None,
) -> StepWiseTrnModel:
    
    trn_cfg = cast(TransformerConfig, model_cfg.params)
    feature_dim = x_train.shape[2]

    # number of time steps to predict 
    t_out = y_train.shape[1]

    # number of time steps to give in input 
    t_in = x_train.shape[1]
    
    model = StepWiseTrnModel(
        feature_dim = feature_dim,
        dim_model = trn_cfg.dim_model,
        num_heads = trn_cfg.num_heads,
        ff_dim = trn_cfg.ff_dim,
        num_layers = trn_cfg.num_layers,
        dropout = trn_cfg.dropout,
        start_mode = model_cfg.inference.start_mode,
        prediction_mode_id = prediction_mode_id[prediction_mode],
        t_out = t_out,
        t_in = t_in
    )

    model.build(tf.TensorShape([None, t_in, feature_dim]))

    optimizer = build_optimizer(
        model_cfg.compile.optimizer,
        model_cfg.compile.learning_rate
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
