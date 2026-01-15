from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, cast

import tensorflow as tf
import keras
from keras import layers

from ..strategies.base_strategy import TrainingStrategy
from ..enums.start_mode import StartMode


class LSTM2LayerTFState(NamedTuple):
    h1: tf.Tensor
    c1: tf.Tensor
    h2: tf.Tensor
    c2: tf.Tensor


class Seq2SeqLSTM2LayerStepWiseModel(keras.Model):
    def __init__(self, *, feature_dim: int, latent_dim: int, start_mode: StartMode):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.start_mode = start_mode

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
        self.ctx_horizon = tf.Variable(tf.constant(0,tf.int32), trainable=False)

    def build(self, input_shape):
        # input_shape: (batch, Tin, D)
        feature_dim = input_shape[-1]
        if feature_dim is None:
            feature_dim = self.feature_dim

        # Encoder build
        self.enc_lstm_1.build((None, None, feature_dim))          # (N,Tin,D) -> (N,Tin,latent)
        self.enc_lstm_2.build((None, None, self.latent_dim))      # (N,Tin,latent)

        # Decoder build (step-wise input: (N,1,D))
        self.dec_lstm_1.build((None, 1, feature_dim))             # (N,1,D) -> (N,1,latent)
        self.dec_lstm_2.build((None, 1, self.latent_dim))         # (N,1,latent)

        # Dense build (last dim = latent)
        self.out_dense.build((None, 1, self.latent_dim))

        super().build(input_shape)

    def set_context(self, *, strategy: TrainingStrategy, epoch: int, total_epochs: int, horizon :int) -> None:
        self._strategy = strategy
        self.ctx_epoch.assign(int(epoch))
        self.ctx_total_epochs.assign(int(total_epochs))
        self.ctx_horizon.assign(int(horizon))

    def _encode(self, X: tf.Tensor) -> LSTM2LayerTFState:
        x = self.enc_lstm_1(X)
        _, h, c = self.enc_lstm_2(x)
        return LSTM2LayerTFState(h1=h, c1=c, h2=h, c2=c)

    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:
        if self.start_mode == StartMode.LAST_X:
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


        T_eff = self.ctx_horizon

        with tf.GradientTape() as tape:

            # return the internal states from encoder 
            state = self._encode(X)

            # 
            dec_t = self._init_dec0(X)

            # array with T_out elements and each element have shape(batch_size, feature_dim)
            # batch_size is None because is dinamic 
            ta = tf.TensorArray(
                dtype=Y.dtype,
                size=T_eff,
                element_shape=tf.TensorShape([None, self.feature_dim]),
            )
            
            # ta = shape (batch_size , T_out , feature_dim)
            # print(ta.element_shape) = shape (batch_size, feature_dim) ,  ta.size() = T_out

            t0 = tf.constant(0, tf.int32)

            strategy = self._strategy  

            if strategy is None:
                raise RuntimeError("strategy not set")

            def cond(t, dec_t, state, ta):
                return t < T_eff

            def body(t, dec_t, state, ta):
                y_t, state = self._decode_step(dec_t, state)    # (N,1,D)
                y_t_nd = tf.squeeze(y_t, axis=1)                # (N,D) 

                ta = ta.write(t, y_t_nd)
                
                # Y.shape(batch_size, timesteps, feature_dim)
                y_true_nd = tf.gather(Y, t, axis=1)             # (N,D)
                y_true_t = tf.expand_dims(y_true_nd, axis=1)    # (N,1,D)

                dec_t = strategy.next_dec_input(
                    y_true_t=y_true_t,
                    y_pred_t=y_t,
                    epoch=tf.convert_to_tensor(self.ctx_epoch),
                    total_epochs=tf.convert_to_tensor(self.ctx_total_epochs),
                )
                return t + 1, dec_t, state, ta

            # the parameter given in input change in the loop 
            # the constant parameter are taken by the father function
            _, _, _, ta = tf.while_loop(cond, body, [t0, dec_t, state, ta], parallel_iterations=1)

            # tf.print(ta.stack())
            # stack: (T_out, N, D) -> transpose: (N, T_out, D)
            Y_pred = tf.transpose(ta.stack(), [1, 0, 2])

            # Y_true: (N, T_eff, D)
            Y_true = tf.slice(Y, [0, 0, 0], [-1, T_eff, -1])

            # loss = self.compiled_loss(Y, Y_pred, regularization_losses=self.losses)
            loss = cast(tf.Tensor, self.compute_loss(x=X, y=Y_true, y_pred=Y_pred, training=True))
                
        # calculate the derivate of the loss from all trainable_variables (weights ect..)
        grads_raw = tape.gradient(loss, self.trainable_variables)

        if grads_raw is None:
            raise RuntimeError("tape.gradient return None")

        grads = cast(Sequence[Optional[tf.Tensor]], grads_raw)

        vars_ = list(self.trainable_variables)

        # apply_gradients don't accept a list of None so we filter it
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
       
        if not grads_and_vars:
            raise RuntimeError("There are no gradients")

        grads, vars_ = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, self.current_clip_norm)
        self.optimizer.apply_gradients(zip(grads, vars_))

        self.compute_metrics(x=X, y=Y_true, y_pred=Y_pred, sample_weight=None)

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

        ta = tf.TensorArray(
            dtype=Y.dtype, size=T_out,  
            element_shape=tf.TensorShape([None, self.feature_dim]),
        )
        
        t0 = tf.constant(0, dtype=tf.int32)

        def cond(t, dec_t, state, ta):
            return t < T_out

        def body(t, dec_t, state, ta):
            y_t, state = self._decode_step(dec_t, state)
            y_t_nd = tf.squeeze(y_t, axis=1)
            ta = ta.write(t, y_t_nd)

            y_true = tf.gather(Y, t, axis=1)
            y_true_t = tf.expand_dims(y_true, axis=1)

           
            return t + 1, y_true_t, state, ta



        _, _, _, ta = tf.while_loop(cond, body, [t0, dec_t, state, ta], parallel_iterations=1)

        preds = tf.transpose(ta.stack(), [1, 0, 2])

        loss = self.compute_loss(x=X, y=Y, y_pred=preds, training=False)
        self.compute_metrics(x=X, y=Y, y_pred=preds, sample_weight=None)

        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        return out
