from pathlib import Path
from typing import Optional, cast


from .step_wise_model import StepWiseTrnModel
from ..model.transformer_config import TransformerConfig
from ..model.model_config import ModelConfig
from ..registry import register_model

from .struct import *

from ..enums.model_type import ModelType
from ..enums.model_variant import ModelVariant
from ..enums.decoder_mode import DecoderMode

from ..core.core import build_optimizer

@register_model(ModelType.TRN, ModelVariant.SEQ2SEQ, DecoderMode.STEP_WISE)
def build_transformer_step_wise_model(
    x_train,
    y_train,
    model_cfg: ModelConfig,
    model_path: Optional[str] = None,
) -> StepWiseTrnModel:
    tcfg = cast(TransformerConfig, model_cfg.params)
    feature_dim = int(x_train.shape[2])

    model = StepWiseTrnModel(
        feature_dim=feature_dim,
        d_model=tcfg.d_model,
        n_heads=tcfg.num_heads,
        ff_dim=tcfg.ff_dim,
        num_layers=tcfg.num_layers,
        dropout=tcfg.dropout,
        max_len=tcfg.max_len,
        start_mode=model_cfg.inference.start_mode,
    )

    model.build((None, None, feature_dim))

    optimizer = build_optimizer(
        model_cfg.compile.optimizer,
        model_cfg.compile.learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss=model_cfg.compile.loss,
        metrics=model_cfg.compile.metrics,
        run_eagerly=model_cfg.compile.run_eagerly,
    )

    if model_path is not None:
        model.load_weights(Path(model_path) / "model.weights.h5")

    return model
