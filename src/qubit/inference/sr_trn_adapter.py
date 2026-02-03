from __future__ import annotations
import tensorflow as tf

from ..models.trn.sr_trn_model import SrTrnModel

class SrTrnAdapter(tf.keras.Model):
    def __init__(self, sr_model: SrTrnModel, name: str = "sr_adapter"):
        super().__init__(name=name)
        self.sr_model = sr_model

    def call(self, X, training: bool = False):
        return self.sr_model(X, training=training)
