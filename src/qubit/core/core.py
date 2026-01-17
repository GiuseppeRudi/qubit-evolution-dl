import tensorflow as tf
from typing import Optional

def build_optimizer(name: str, lr: float, clip_norm: Optional[float] = None):
    name = name.lower()

    kwargs = {"learning_rate": lr}
    if clip_norm is not None and clip_norm > 0:
        kwargs["global_clipnorm"] = clip_norm  

    if name == "adam":
        return tf.keras.optimizers.Adam(**kwargs)
    elif name == "adamw":
        return tf.keras.optimizers.AdamW(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")