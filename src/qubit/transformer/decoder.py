import tensorflow as tf
import keras
from keras import layers

class DecoderTRN(layers.Layer):
    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int, dropout: float, name: str):
        super().__init__(name=name)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")
        self.self_mha = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
            name=f"{name}_self_mha",
        )
        self.drop1 = layers.Dropout(dropout, name=f"{name}_drop1")

        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")
        self.cross_mha = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
            name=f"{name}_cross_mha",
        )
        self.drop2 = layers.Dropout(dropout, name=f"{name}_drop2")

        self.ln3 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln3")
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ],
            name=f"{name}_ffn",
        )
        self.drop3 = layers.Dropout(dropout, name=f"{name}_drop3")

    def call(
        self,
        x: tf.Tensor,
        *,
        memory: tf.Tensor,
        causal_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        # masked self-attn (causal)
        y = self.ln1(x)
        attn1 = self.self_mha(query=y, value=y, key=y, attention_mask=causal_mask, training=training)
        x = x + self.drop1(attn1, training=training)

        # cross-attn (decoder queries -> encoder memory)
        y = self.ln2(x)
        attn2 = self.cross_mha(query=y, value=memory, key=memory, training=training)
        x = x + self.drop2(attn2, training=training)

        # FFN
        y = self.ln3(x)
        f = self.ffn(y, training=training)
        x = x + self.drop3(f, training=training)
        return x
