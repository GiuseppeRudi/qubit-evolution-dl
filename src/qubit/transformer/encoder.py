import tensorflow as tf
import keras
from keras import layers

class EncoderTRN(layers.Layer):
    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int, dropout: float, name: str):
        super().__init__(name=name)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")
        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
            name=f"{name}_mha",
        )
        self.drop1 = layers.Dropout(dropout, name=f"{name}_drop1")

        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ],
            name=f"{name}_ffn",
        )
        self.drop2 = layers.Dropout(dropout, name=f"{name}_drop2")

    def call(self, x: tf.Tensor, *, training: bool) -> tf.Tensor:
        # PreNorm self-attn
        y = self.ln1(x)
        attn = self.mha(query=y, value=y, key=y, training=training)  # no mask for encoder
        x = x + self.drop1(attn, training=training)

        # PreNorm FFN
        y = self.ln2(x)
        f = self.ffn(y, training=training)
        x = x + self.drop2(f, training=training)
        return x

