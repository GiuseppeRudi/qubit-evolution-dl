from __future__ import annotations
from dataclasses import dataclass
import tensorflow as tf
import keras
from keras import layers
from typing import Optional, Sequence, cast
   

from ..strategies.base_strategy import TrainingStrategy

@dataclass
class LSTM2LayerTFState:
    h1: tf.Tensor
    c1: tf.Tensor
    h2: tf.Tensor
    c2: tf.Tensor


class Seq2SeqLSTM2LayerStepWiseModel(keras.Model):
    def __init__(self, *, feature_dim: int, latent_dim: int, start_mode: str = "zeros"):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.start_mode = start_mode

        # Encoder
        self.enc_lstm_1 = layers.LSTM(latent_dim, return_sequences=True, name="enc_lstm_1")
        self.enc_lstm_2 = layers.LSTM(latent_dim, return_state=True, name="enc_lstm_2")

        # Decoder (usati step-by-step con input (N,1,D))
        self.dec_lstm_1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm_1")
        self.dec_lstm_2 = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm_2")

        self.out_dense = layers.Dense(feature_dim, name="out_dense")

        # contesto “runtime” settato dal trainer
        self._strategy: TrainingStrategy | None = None
        self._ctx_epoch: int = 0
        self._ctx_total_epochs: int = 1

    def set_context(self, *, strategy: TrainingStrategy, epoch: int, total_epochs: int) -> None:
        self._strategy = strategy
        self._ctx_epoch = int(epoch)
        self._ctx_total_epochs = int(total_epochs)

    def _encode(self, X: tf.Tensor) -> LSTM2LayerTFState:
        x = self.enc_lstm_1(X)
        _, h, c = self.enc_lstm_2(x)
        # coerente con il tuo training full-seq (passi enc_states anche al secondo decoder)
        return LSTM2LayerTFState(h1=h, c1=c, h2=h, c2=c)

    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:
        # X: (N, T_in, D)

        if self.start_mode == "last_x":
            # TODO: change the shape
            T = tf.gather(tf.shape(X), 1)          # T_in
            last = tf.gather(X, T - 1, axis=1)     # (N, D)
            return tf.expand_dims(last, axis=1)    # (N, 1, D)

        N = X.shape[0]
        D = X.shape[2]

        dtype = X.dtype
        if dtype is None:          # solo per far contento Pylance (a runtime non succede)
            dtype = tf.float32
   
        return tf.zeros((N, 1, D), dtype=dtype)


    def _decode_step(self, dec_t: tf.Tensor, state: LSTM2LayerTFState) -> tuple[tf.Tensor, LSTM2LayerTFState]:
        # dec_t: (N,1,D)
        dec_seq1, h1_out, c1_out = self.dec_lstm_1(dec_t, initial_state=[state.h1, state.c1])
        dec_seq2, h2_out, c2_out = self.dec_lstm_2(dec_seq1, initial_state=[state.h2, state.c2])
        y_t = self.out_dense(dec_seq2)  # (N,1,D)
        new_state = LSTM2LayerTFState(h1=h1_out, c1=c1_out, h2=h2_out, c2=c2_out)
        return y_t, new_state

    def train_step(self, data):
        X, Y = data  # X:(N,Tin,D)  Y:(N,Tout,D)

        print("static:", X.shape, Y.shape)
        

        if self._strategy is None:
            raise RuntimeError("StepWise model: strategy context not set. Call model.set_context(...) before fit().")

        T_out = Y.shape[1]

        with tf.GradientTape() as tape:
            state = self._encode(X)
            dec_t = self._init_dec0(X)

            preds = []
            for t in range(T_out):
                y_t, state = self._decode_step(dec_t, state)
                preds.append(y_t)

                y_true_t = Y[:, t:t+1, :]  # (N,1,D)
                dec_t = self._strategy.next_dec_input(
                    y_true_t=y_true_t,
                    y_pred_t=y_t,
                    epoch=self._ctx_epoch,
                    total_epochs=self._ctx_total_epochs,
                )

            Y_pred = tf.concat(preds, axis=1)  # (N,Tout,D)
            loss = self.compiled_loss(Y, Y_pred, regularization_losses=self.losses)


        grads_raw = tape.gradient(loss, self.trainable_variables)

        if grads_raw is None:
            raise RuntimeError("tape.gradient ha restituito None (loss non dipende dalle variabili?)")

        # Dici a Pylance cosa ti aspetti davvero
        grads = cast(Sequence[Optional[tf.Tensor]], grads_raw)

        vars_ = list(self.trainable_variables)  # rende l’iterabile “concreto” per Pylance

        grads_and_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if not grads_and_vars:
            raise RuntimeError("Nessun gradiente valido: controlla la loss e il grafo del modello.")

        self.optimizer.apply_gradients(grads_and_vars)
        self.compiled_metrics.update_state(Y, Y_pred)


        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        return out

    def test_step(self, data):
        # per validation, usa la stessa strategy/contesto impostata dal trainer (fase corrente)
        X, Y = data
        if self._strategy is None:
            raise RuntimeError("StepWise model: strategy context not set. Call model.set_context(...) before fit().")

        T_out = tf.gather(tf.shape(Y), 1)
        state = self._encode(X)
        dec_t = self._init_dec0(X)

        preds = []
        for t in tf.range(T_out):
            y_t, state = self._decode_step(dec_t, state)
            preds.append(y_t)

            y_true_t = Y[:, t:t+1, :]
            dec_t = self._strategy.next_dec_input(
                y_true_t=y_true_t,
                y_pred_t=y_t,
                epoch=self._ctx_epoch,
                total_epochs=self._ctx_total_epochs,
            )

        Y_pred = tf.concat(preds, axis=1)
        loss = self.compiled_loss(Y, Y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(Y, Y_pred)

        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        return out
