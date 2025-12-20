from typing import cast

from ..model.transformer_config import TransformerConfig
from ..model.model_config import ModelConfig
from ..registry import register_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
from .struct import *


@register_model("TRN", "ENCODERDENSE")
def build_transformer_model(
    x_train,
    y_train,
    model_cfg: ModelConfig
):
    params = cast(TransformerConfig,model_cfg.params)

    embed_dim=params.embed_dim
    num_heads=params.num_heads
    ff_dim=params.ff_dim

    # x_train: (N, in_len, feat)
    # y_train: (N, out_len, feat)
    if x_train.ndim != 3 or y_train.ndim != 3:
        raise ValueError(f"Expected 3D tensors. Got x:{x_train.shape}, y:{y_train.shape}")

    input_seq_len, x_feat = x_train.shape[1], x_train.shape[2]
    output_seq_len, y_feat = y_train.shape[1], y_train.shape[2]

    if x_feat != y_feat:
        raise ValueError(f"Feature dim mismatch: x has {x_feat}, y has {y_feat}")

    input_shape = (input_seq_len, x_feat)
    feature_dim = x_feat

    inputs = Input(shape=input_shape)

    x = Dense(embed_dim, activation="relu")(inputs)

    x = PositionalEmbedding(input_seq_len, input_seq_len, embed_dim)(x)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x, training=False)

    x = GlobalAveragePooling1D()(x)

    x = Dense(output_seq_len * feature_dim, activation="linear")(x)

    outputs = Reshape((output_seq_len, feature_dim))(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_cfg.name)
    model.compile(optimizer=model_cfg.compile.optimizer, loss=model_cfg.compile.loss)
    return model
