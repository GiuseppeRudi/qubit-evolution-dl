import tensorflow as tf
from typing import Optional

def build_optimizer(name: str, lr: float, clip_norm: Optional[float] = None):
    name = name.lower()

    kwargs = {"learning_rate": lr}
    if isinstance(lr, str):
        kwargs["learning_rate"] = float(lr)
    if clip_norm is not None and clip_norm > 0:
        kwargs["global_clipnorm"] = clip_norm  

    if name == "adam": return tf.keras.optimizers.Adam(**kwargs)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(**kwargs)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(**kwargs)
    else: raise ValueError(f"Unknown optimizer: {name}")