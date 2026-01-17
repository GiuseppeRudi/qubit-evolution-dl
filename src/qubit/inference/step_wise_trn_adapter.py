from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from ..enums.start_mode import StartMode
from ..enums.verbose_mode import VerboseMode
from .base import AutoregressiveAdapter  # il tuo

@dataclass
class TransformerStepWiseNPState:
    memory: np.ndarray   # (B, Tin, d_model)
    prefix: np.ndarray   # (B, L, D)  -> decoder inputs già consumati (D = feature_dim)


class StepWiseTrnAdapter(AutoregressiveAdapter):
    def __init__(self, model, *, verbose: VerboseMode):
        self.model = model
        self.verbose = verbose

    @property
    def feature_dim(self) -> int:
        return int(self.model.feature_dim)

    def encode(self, X: np.ndarray, *, batch_size: int) -> TransformerStepWiseNPState:
        Xtf = tf.convert_to_tensor(X, dtype=tf.float32)

        # IMPORTANT: il tuo model deve esporre _encode(X, training=...)
        mem_tf = self.model._encode(Xtf, training=False)  # (B, Tin, d_model)
        mem = mem_tf.numpy().astype(np.float32, copy=False)

        # prefix inizialmente vuota: (B, 0, D)
        B = X.shape[0]
        D = self.feature_dim
        prefix0 = np.zeros((B, 0, D), dtype=np.float32)

        return TransformerStepWiseNPState(memory=mem, prefix=prefix0)

    def init_decoder_input(self, X: np.ndarray, *, start_mode: StartMode) -> np.ndarray:
        # stesso comportamento del tuo LSTM adapter
        self.model.start_mode = start_mode
        Xtf = tf.convert_to_tensor(X, dtype=tf.float32)

        # IMPORTANT: il tuo model deve esporre _init_dec0(X)
        dec0 = self.model._init_dec0(Xtf)  # (B,1,D)
        return dec0.numpy().astype(np.float32, copy=False)

    def step(
        self,
        dec_t: np.ndarray,
        state: TransformerStepWiseNPState,
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, TransformerStepWiseNPState]:
        """
        dec_t: (B,1,D) input al decoder per questo step
        state.memory: (B,Tin,d_model)
        state.prefix: (B,L,D)  (decoder inputs già passati)
        """

        dec_tf = tf.convert_to_tensor(dec_t, dtype=tf.float32)
        mem_tf = tf.convert_to_tensor(state.memory, dtype=tf.float32)
        pref_tf = tf.convert_to_tensor(state.prefix, dtype=tf.float32)

        # append input corrente alla prefix
        new_pref = tf.concat([pref_tf, dec_tf], axis=1)  # (B, L+1, D)

        # safety: evitare overflow pos-embedding (se hai max_len piccolo)
        # L = tf.shape(new_pref)[1]
        # tf.debugging.assert_less(L, tf.cast(self.model.max_len, tf.int32))

        # IMPORTANT: il tuo model deve esporre _decode_prefix(prefix, memory, training=...)
        y_seq = self.model._decode_prefix(new_pref, mem_tf, training=False)  # (B, L+1, D)
        y_t = y_seq[:, -1:, :]  # (B,1,D)

        y_np = y_t.numpy().astype(np.float32, copy=False)
        new_state = TransformerStepWiseNPState(
            memory=state.memory,                 # invariata
            prefix=new_pref.numpy().astype(np.float32, copy=False),
        )
        return y_np, new_state
