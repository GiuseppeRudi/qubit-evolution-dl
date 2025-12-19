from ..registry import register_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
from .struct import *


@register_model("TRN", "ENCODERDENSE")
def build_transformer_model(
    x_train,
    y_train,
    model_cfg
):
    
    embed_dim=64
    num_heads=4
    ff_dim=128
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

    # 1) Proiezione alla dimensione embedding
    x = Dense(embed_dim, activation="relu")(inputs)

    # 2) Positional embedding (come nel tuo codice)
    x = PositionalEmbedding(input_seq_len, input_seq_len, embed_dim)(x)

    # 3) Blocco Transformer (Encoder)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x, training=False)

    # 4) Pooling -> vettore contesto
    x = GlobalAveragePooling1D()(x)

    # 5) "Decoder" denso per regressione
    x = Dense(output_seq_len * feature_dim, activation="linear")(x)

    # 6) reshape in (out_len, feat)
    outputs = Reshape((output_seq_len, feature_dim))(x)

    model = Model(inputs=inputs, outputs=outputs, name="Transformer_Model")
    model.compile(optimizer="adam", loss="mse")
    return model
