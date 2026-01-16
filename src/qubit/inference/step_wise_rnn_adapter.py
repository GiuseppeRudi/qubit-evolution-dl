from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from ..enums.start_mode import StartMode

from .base import AutoregressiveAdapter
from ..rnn.Seq2SeqLSTM2LayerStepWiseModel import LSTM2LayerTFState

from ..enums.verbose_mode import VerboseMode

@dataclass
class LSTM2LayerNPState:
    h1: np.ndarray
    c1: np.ndarray
    h2: np.ndarray
    c2: np.ndarray


class StepWiseSeq2SeqAdapter(AutoregressiveAdapter):
    def __init__(self, model, *, verbose: VerboseMode ):
        self.model = model

        # TODO verify if the verbose must send to the  model functions
        self.verbose = verbose

    @property
    def feature_dim(self) -> int:
        return int(self.model.feature_dim)

    def encode(self, X: np.ndarray, *, batch_size: int) -> LSTM2LayerNPState:
        # converti in TF e usa i layer del model
        Xtf = tf.convert_to_tensor(X, dtype=tf.float32)
        st = self.model._encode(Xtf)  # LSTM2LayerTFState

        return LSTM2LayerNPState(
            h1=st.h1.numpy(), c1=st.c1.numpy(),
            h2=st.h2.numpy(), c2=st.c2.numpy(),
        )

    def init_decoder_input(self, X: np.ndarray, *, start_mode: StartMode) -> np.ndarray:
        # print("\nStepWiseSeq2SeqAdapter.init_decoder_input called with start_mode:", start_mode)
       
        self.model.start_mode = start_mode 
        Xtf = tf.convert_to_tensor(X, dtype=tf.float32)
        dec0 = self.model._init_dec0(Xtf)  # (N,1,D)
        return dec0.numpy().astype(np.float32, copy=False)

    def step(
        self,
        dec_t: np.ndarray,
        state: LSTM2LayerNPState,
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, LSTM2LayerNPState]:

        dec_tf = tf.convert_to_tensor(dec_t, dtype=tf.float32)

        st_tf = LSTM2LayerTFState(
            h1=tf.convert_to_tensor(state.h1, dtype=tf.float32),
            c1=tf.convert_to_tensor(state.c1, dtype=tf.float32),
            h2=tf.convert_to_tensor(state.h2, dtype=tf.float32),
            c2=tf.convert_to_tensor(state.c2, dtype=tf.float32),
        )

        y_t, new_st = self.model._decode_step(dec_tf, st_tf)

        return (
            y_t.numpy().astype(np.float32, copy=False),
            LSTM2LayerNPState(
                h1=new_st.h1.numpy(), c1=new_st.c1.numpy(),
                h2=new_st.h2.numpy(), c2=new_st.c2.numpy(),
            )
        )
