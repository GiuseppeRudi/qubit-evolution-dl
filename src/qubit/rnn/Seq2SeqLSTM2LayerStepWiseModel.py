from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, cast

import tensorflow as tf
import keras
from keras import layers

from ..strategies.base_strategy import TrainingStrategy


class LSTM2LayerTFState(NamedTuple):
    h1: tf.Tensor
    c1: tf.Tensor
    h2: tf.Tensor
    c2: tf.Tensor


class Seq2SeqLSTM2LayerStepWiseModel(keras.Model):
    def __init__(self, *, feature_dim: int, latent_dim: int, start_mode: str = "zeros"):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.start_mode = str(start_mode)

        # Encoder
        self.enc_lstm_1 = layers.LSTM(latent_dim, return_sequences=True, name="enc_lstm_1")
        self.enc_lstm_2 = layers.LSTM(latent_dim, return_state=True, name="enc_lstm_2")

        # Decoder (step-by-step, input (N,1,D))
        self.dec_lstm_1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm_1")
        self.dec_lstm_2 = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm_2")

        self.out_dense = layers.Dense(feature_dim, name="out_dense")

        # Runtime strategy
        self._strategy: Optional[TrainingStrategy] = None

        # Epoch context as TF vars (evita retracing in graph)
        self.ctx_epoch = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False)
        self.ctx_total_epochs = tf.Variable(tf.constant(1, dtype=tf.int32), trainable=False)

    def set_context(self, *, strategy: TrainingStrategy, epoch: int, total_epochs: int) -> None:
        self._strategy = strategy
        self.ctx_epoch.assign(int(epoch))
        self.ctx_total_epochs.assign(int(total_epochs))

    def _encode(self, X: tf.Tensor) -> LSTM2LayerTFState:
        x = self.enc_lstm_1(X)
        _, h, c = self.enc_lstm_2(x)
        return LSTM2LayerTFState(h1=h, c1=c, h2=h, c2=c)

    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:
        if self.start_mode == "last_x":
            T = tf.shape(X)[1]
            last = tf.gather(X, T - 1, axis=1)   # (N, D)
            return tf.expand_dims(last, axis=1)  # (N, 1, D)

        N = tf.shape(X)[0]
        D = self.feature_dim         
        return tf.zeros(tf.stack([N, 1, D]), dtype=X.dtype)

    def _decode_step(self, dec_t: tf.Tensor, state: LSTM2LayerTFState) -> tuple[tf.Tensor, LSTM2LayerTFState]:
        dec_seq1, h1_out, c1_out = self.dec_lstm_1(dec_t, initial_state=[state.h1, state.c1])
        dec_seq2, h2_out, c2_out = self.dec_lstm_2(dec_seq1, initial_state=[state.h2, state.c2])
        y_t = self.out_dense(dec_seq2)  # (N,1,D)
        return y_t, LSTM2LayerTFState(h1=h1_out, c1=c1_out, h2=h2_out, c2=c2_out)

    def train_step(self, data):
        X, Y = data  # X:(N,Tin,D), Y:(N,Tout,D)

        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])


        if self._strategy is None:
            raise RuntimeError("StepWise model: strategy context not set. Call model.set_context(...) before fit().")

        T_out = tf.shape(Y)[1]

        # forza shape attesa (molto utile per beccare mismatch subito)
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        with tf.GradientTape() as tape:
            state = self._encode(X)
            dec_t = self._init_dec0(X)

            # ogni elemento: (N, D)
            ta = tf.TensorArray(
                dtype=Y.dtype,
                size=T_out,
                element_shape=tf.TensorShape([None, self.feature_dim]),
            )
            t0 = tf.constant(0, tf.int32)

            strategy = self._strategy  # narrowing per Pylance
            if strategy is None:
                raise RuntimeError("strategy not set")

            def cond(t, dec_t, state, ta):
                return t < T_out

            def body(t, dec_t, state, ta):
                y_t, state = self._decode_step(dec_t, state)    # (N,1,D)
                y_t_nd = tf.squeeze(y_t, axis=1)                # (N,D)  <--- QUI
                ta = ta.write(t, y_t_nd)

                # y_true_t: (N,1,D)
                y_true_nd = tf.gather(Y, t, axis=1)             # (N,D)
                y_true_t = tf.expand_dims(y_true_nd, axis=1)    # (N,1,D)

                dec_t = strategy.next_dec_input(
                    y_true_t=y_true_t,
                    y_pred_t=y_t,
                    epoch=tf.convert_to_tensor(self.ctx_epoch),
                    total_epochs=tf.convert_to_tensor(self.ctx_total_epochs),
                )
                return t + 1, dec_t, state, ta

            _, _, _, ta = tf.while_loop(cond, body, [t0, dec_t, state, ta], parallel_iterations=1)

            # stack: (T_out, N, D) -> transpose: (N, T_out, D)
            Y_pred = tf.transpose(ta.stack(), [1, 0, 2])
            Y_pred = tf.ensure_shape(Y_pred, [None, None, self.feature_dim])

            # Debug assert (lascia finché non sei sicuro)
            tf.debugging.assert_equal(tf.shape(Y), tf.shape(Y_pred), message="Y vs Y_pred shape mismatch")

            loss = self.compiled_loss(Y, Y_pred, regularization_losses=self.losses)

        grads_raw = tape.gradient(loss, self.trainable_variables)
        if grads_raw is None:
            raise RuntimeError("tape.gradient ha restituito None (loss non dipende dalle variabili?)")

        grads = cast(Sequence[Optional[tf.Tensor]], grads_raw)
        vars_ = list(self.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if not grads_and_vars:
            raise RuntimeError("Nessun gradiente valido: controlla la loss e il grafo del modello.")

        self.optimizer.apply_gradients(grads_and_vars)
        self.compiled_metrics.update_state(Y, Y_pred)

        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        return out

    def test_step(self, data):
        X, Y = data

        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        if self._strategy is None:
            raise RuntimeError("StepWise model: strategy context not set. Call model.set_context(...) before fit().")

        T_out = tf.shape(Y)[1]

        state = self._encode(X)
        dec_t = self._init_dec0(X)

        ta = tf.TensorArray(dtype=Y.dtype, size=T_out)
        t0 = tf.constant(0, dtype=tf.int32)

        def cond(t, dec_t, state, ta):
            return t < T_out

        def body(t, dec_t, state, ta):
            y_t, state = self._decode_step(dec_t, state)
            ta = ta.write(t, y_t)

            y_true = tf.gather(Y, t, axis=1)
            y_true_t = tf.expand_dims(y_true, axis=1)

            dec_t = cast(TrainingStrategy,self._strategy).next_dec_input(
                y_true_t=y_true_t,
                y_pred_t=y_t,
                epoch=self.ctx_epoch,
                total_epochs=self.ctx_total_epochs,
            )
            return t + 1, dec_t, state, ta



        _, _, _, ta = tf.while_loop(cond, body, [t0, dec_t, state, ta], parallel_iterations=1)

        # preds = tf.transpose(ta.stack(), [1, 0, 2])
        preds = tf.transpose(ta.stack(), [1, 0, 2, 3])
        print(  "DEBUG preds shape before squeeze:", preds.shape)
        preds = tf.squeeze(preds, axis=2)

        # preds  = tf.ensure_shape(preds, [None, None, self.feature_dim])

        # # Debug assert (lascia finché non sei sicuro)
        # tf.debugging.assert_equal(tf.shape(Y), tf.shape(preds), message="Y vs Y_pred shape mismatch")

        loss = self.compiled_loss(Y, preds, regularization_losses=self.losses)
        self.compiled_metrics.update_state(Y, preds)

        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        return out
