from pathlib import Path
from typing import Optional, cast

import tensorflow as tf

from .hybrid_trn_model import HybridTrnModel
from .sr_trn_model import SrTrnModel

from ...dataclasses.transformer_config import TransformerConfig
from ...dataclasses.model_config import ModelConfig
from ...dataclasses.sr_config import SuperResolutionConfig
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

@register_model(ModelType.TRN, ModelVariant.SUPER_RESOLUTION, DecoderMode.ENCODER_ONLY)  
def build_transformer_sr_model(
    x_train, # x_train.shape(num_windows, windows_len, feature_dim + 1 (mask channel))
    y_train, # x_train.shape(num_windows, windows_len, feature_dim )
    model_cfg: ModelConfig,
    sr_cfg: SuperResolutionConfig,
    model_path: Optional[str] = None,
) -> SrTrnModel:

    # widows_len == output_seq_len == input_seq_len * sr.stride 

    trn_cfg = cast(TransformerConfig, model_cfg.params)

    # widows_len == output_seq_len == input_seq_len * sr.stride 
    windows_len = int(x_train.shape[1])   
    # input_feature_dim == feature_dim (magnetizzazions and correlations) + 1 (mask channel)          
    input_feature_dim = int(x_train.shape[2])  

    feature_dim = int(y_train.shape[2])         

    model = SrTrnModel(
        feature_dim=feature_dim,
        input_feature_dim=input_feature_dim,
        windows_len=windows_len,
        dim_model=trn_cfg.dim_model,
        num_heads=trn_cfg.num_heads,
        ff_dim=trn_cfg.ff_dim,
        num_layers=trn_cfg.num_layers,
        dropout=trn_cfg.dropout,
        sr_cfg=sr_cfg,
    )


    optimizer = build_optimizer(
        model_cfg.compile.optimizer, 
        model_cfg.compile.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=model_cfg.compile.loss,       
        metrics=model_cfg.compile.metrics, 
        run_eagerly=model_cfg.compile.run_eagerly,
    )

    model.build(tf.TensorShape([None, windows_len, input_feature_dim]))

    if model_path is not None:
        model.load_weights(Path(model_path) / "model.weights.h5")

    return model

@register_model(ModelType.TRN, ModelVariant.FORECASTING, DecoderMode.HYBRID)
def build_transformer_hybrid_model(
    x_train,# X.shape (n_windows , input_seq_len , feature_dim)
    y_train, # Y.shape (n_windows , output_seq_len  , feature_dim)
    model_cfg: ModelConfig,
    prediction_mode: PredictionMode,
    model_path: Optional[str] = None,
) -> HybridTrnModel:
    
    trn_cfg = cast(TransformerConfig, model_cfg.params)
    feature_dim = x_train.shape[2]

    # number of time steps to predict 
    t_out = y_train.shape[1]

    # number of time steps to give in input 
    t_in = x_train.shape[1]
    
    model = HybridTrnModel(
        feature_dim = feature_dim,
        dim_model = trn_cfg.dim_model,
        num_heads = trn_cfg.num_heads,
        ff_dim = trn_cfg.ff_dim,
        num_layers = trn_cfg.num_layers,
        dropout = trn_cfg.dropout,
        start_mode = model_cfg.inference.start_mode,
        prediction_mode_id = prediction_mode_id[prediction_mode],
        t_out = t_out,
        t_in = t_in,
    )


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

    model.build(tf.TensorShape([None, t_in, feature_dim]))

    if model_path is not None:
        model.load_weights(Path(model_path) / "model.weights.h5")

    return model
