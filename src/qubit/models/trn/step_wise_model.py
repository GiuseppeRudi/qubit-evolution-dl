from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, cast

import tensorflow as tf
import keras
from keras import layers

from ...enums.start_mode import StartMode

from .decoder import DecoderTRNBlock
from .encoder import EncoderTRNBlock

from ..strategy_chooser import StrategyChooserModel

from ...utils.layers_names import OUT_DENSE, ENC_IN_PROJ, ENC_IN_DROP, ENC_POS_EMB, ENC_BLOCK_, DEC_IN_PROJ, DEC_IN_DROP, DEC_POS_EMB, DEC_BLOCK_

class StepWiseTrnModel(StrategyChooserModel):
    def __init__(
        self,
        *,
        feature_dim: int,
        dim_model: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
        start_mode: StartMode,
        prediction_mode_id: int,
        t_out: int,
        t_in: int
    ):
        super().__init__(t_out=t_out,prediction_mode_id=prediction_mode_id)
        self.feature_dim = feature_dim
        self.dim_model = dim_model # internal dimension of vector  
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.start_mode = start_mode
        self.t_in = t_in
        
        # Projections (continuous features -> dim_model)
        # needed because the internal rappresentations of transformer
        # work with dim_model as third dimension and not feature_dim
        self.enc_in = layers.Dense(dim_model, name = ENC_IN_PROJ)
        self.dec_in = layers.Dense(dim_model, name = DEC_IN_PROJ)

        # Positional embeddings 
        # The transformer architecture doesn't know the order of sequence in input
        # instead the lstm use the second dimension automatically (timesteps)
        # For this reason we need to use the Embedding layer to create a matrix of weights (input_dim ,output_dim)
        # where there are input_dim vectors, each with size output_dim that contain the trainable weights

        # enc_pos 
        # input  => pos_idx.shape(input_seq_len,) => list of index from 0 to input_seq_len - 1
        # output => pos_emb.shape(input_seq_len, dim_model) => pos_emb[tf.newaxis, : ,: ].shape(1, input_seq_len, dim_model)
        self.enc_pos = layers.Embedding(input_dim = t_in, output_dim = dim_model, name = ENC_POS_EMB)
        
        # dec_pos 
        # input  => pos_idx.shape(output_seq_len) => list of index from 0 to output_seq_len - 1
        # output => pos_emb.shape(output_seq_len, dim_model) => pos_emb[tf.newaxis, : ,: ].shape(1, output_seq_len, dim_model)
        self.dec_pos = layers.Embedding(input_dim = t_out, output_dim = dim_model, name = DEC_POS_EMB)

        # dropout layer needed to reduce the overfitting and increase the generalizations 
        # putting some values of the tensor given in input to zero with a dropout ratio
        # To activate it takes in input the flag training = true

        # dropout layer don't change the shape so the input.shape == output.shape
        self.enc_drop = layers.Dropout(dropout, name = ENC_IN_DROP)
        self.dec_drop = layers.Dropout(dropout, name = DEC_IN_DROP)

        # unlike the LSTM model where we use directly the layers.LSTM 
        # in the transformer architecture we don't have a unique layer
        # but we work with Block (encoder or decoder) that is used inside multi layers to create
        # a structure using (self.attn + ffn) with layerNorm + residuals + dropout 
        self.enc_blocks = [
            EncoderTRNBlock(dim_model = dim_model, num_heads = num_heads, ff_dim = ff_dim, dropout = dropout, name = f"{ENC_BLOCK_}{i}")
            for i in range(num_layers)
        ]

        self.dec_blocks = [
            DecoderTRNBlock(dim_model = dim_model, num_heads = num_heads, ff_dim = ff_dim, dropout = dropout, name = f"{DEC_BLOCK_}{i}")
            for i in range(num_layers)
        ]

        # input => (batch_size, T, d_model)
        # output => (batch_size, T, feature_dim)
        self.out_dense = layers.Dense(feature_dim, name = OUT_DENSE)
        
        self.ctx_epoch = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False)
        self.ctx_total_epochs = tf.Variable(tf.constant(1, dtype=tf.int32), trainable=False)
        self.ctx_horizon = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def build(self, input_shape: tf.TensorShape):
        
        # input_shape => X.shape(batch_size,input_seq_len, feature_dim)
        
        # self.enc_in => layers.Dense 
        # input => X.shape(batch_size, input_seq_len,feature_dim)
        # output => (batch_size, input_seq_len, dim_model)
        self.enc_in.build(input_shape)

        # self.dec_in => layers.Dense
        # input  => dec_in.shape (batch_size, T, feature_dim)
        # output =>            (batch_size, T, dim_model)
        # where:
        # - FULL_SEQ:  T = output_seq_len (t_out)
        # - STEP_WISE (prefix growing): T starts from 1 and increases each step (1, 2, 3, ..., t_out)
        #   because we decode the whole prefix at every iteration and take the last timestep as current prediction.
        self.dec_in.build((None, None, self.feature_dim))

        # self.out_dense => layers.Dense 
        # input => y_pred.shape(batch_size, T,dim_model)
        # output => (batch_size, T, feature_dim)
        # if train_dec_mode_id == 0 (FULL_SEQ) where T = output_seq_len
        # if train_dec_mode_id == 1 (STEP_WISE) where T = 1
        self.out_dense.build((None, None, self.dim_model))

        tf.print("enc_in output_shape:", self.enc_in.compute_output_shape(input_shape))
        tf.print("dec_in output_shape:", self.dec_in.compute_output_shape((None, None, self.feature_dim)))
        tf.print("out_dense output_shape:", self.out_dense.compute_output_shape((None, None, self.dim_model)))

        tf.print(self.enc_pos)
        tf.print(self.dec_pos)
        tf.print(self.enc_drop)
        tf.print(self.dec_drop)
        tf.print(self.enc_blocks)
        tf.print(self.dec_blocks)

        super().build(input_shape)

    def _causal_mask(self, L: tf.Tensor) -> tf.Tensor:
        # returns (1, L, L) boolean lower-triangular mask
        ones = tf.ones((L, L), dtype=tf.float32)
        band = tf.linalg.band_part(ones, -1, 0)  # lower triangle incl diag
        mask = tf.cast(band > 0.0, tf.bool)
        return mask[tf.newaxis, :, :]  # (1,L,L), broadcast over batch :contentReference[oaicite:1]{index=1}

    def _encode(self, X: tf.Tensor, *, training: bool) -> tf.Tensor:
        # X: (B,Tin,D) -> memory: (B,Tin,dim_model)
        B = tf.shape(X)[0]
        Tin = tf.shape(X)[1]

        x = self.enc_in(X)
        pos = tf.range(Tin, dtype=tf.int32)
        pos_emb = self.enc_pos(pos)[tf.newaxis, :, :]  # (1,Tin,dim_model)
        x = x + pos_emb
        x = self.enc_drop(x, training=training)

        for blk in self.enc_blocks:
            x = blk(x, training=training)

        # memory
        x = tf.ensure_shape(x, [None, None, self.dim_model])
        return x

    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:
        # returns (B,1,D)
        if self.start_mode == StartMode.LAST_X:
            T = tf.shape(X)[1]
            last = tf.gather(X, T - 1, axis=1)   # (B, D)
            return tf.expand_dims(last, axis=1)  # (B, 1, D)

        B = tf.shape(X)[0]
        D = self.feature_dim
        return tf.zeros(tf.stack([B, 1, D]), dtype=X.dtype)

    def _decode_prefix(self, dec_prefix: tf.Tensor, memory: tf.Tensor, *, training: bool) -> tf.Tensor:
        # dec_prefix: (B,L,D) -> y_seq: (B,L,D)
        L = tf.shape(dec_prefix)[1]

        y = self.dec_in(dec_prefix)  # (B,L,dim_model)
        pos = tf.range(L, dtype=tf.int32)
        pos_emb = self.dec_pos(pos)[tf.newaxis, :, :]  # (1,L,dim_model)
        y = y + pos_emb
        y = self.dec_drop(y, training=training)

        causal = self._causal_mask(L)  # (1,L,L)

        for blk in self.dec_blocks:
            y = blk(y, memory=memory, causal_mask=causal, training=training)

        out = self.out_dense(y)  # (B,L,D)
        return out

    
    @property
    def metrics(self):
        # come nel LSTM: una metrica loss manuale
        return [self.loss_tracker]

    def train_step(self, data):
        X, Y = data  # X:(B,Tin,D), Y:(B,Tout,D)
        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        T_out = tf.convert_to_tensor(self.rt.t_out)

        with tf.GradientTape() as tape:
            memory = self._encode(X, training=True) # (B,Tin,dim_model)
            dec_prefix = self._init_dec0(X) # (B,1,D)

            ta = tf.TensorArray(
                dtype=Y.dtype,
                size=T_out,
                element_shape=tf.TensorShape([None, self.feature_dim]),
            )

            t0 = tf.constant(0, tf.int32)

            def cond(t, dec_prefix, ta):
                return t < T_out

            def body(t, dec_prefix, ta):
                # decode full prefix -> prendi ultimo passo come pred corrente
                y_seq = self._decode_prefix(dec_prefix, memory, training=True) # (B,L,D)
                y_t   = y_seq[:, -1:, :] # (B,1,D)
                ta = ta.write(t, tf.squeeze(y_t, axis=1)) # (B,D)

                y_true_t = Y[:, t:t+1, :] # (B,1,D)

                # >>> QUI come LSTM: scegli next input via phase_id + switch_case
                next_in = self.apply_strategy_step_wise(
                    y_true_t=y_true_t,
                    y_pred_t=y_t,
                )

                # forza shape (B,1,D)
                next_in = next_in[:, :1, :]
                next_in = tf.ensure_shape(next_in, [None, 1, self.feature_dim])

                # append al prefix
                dec_prefix = tf.concat([dec_prefix, next_in], axis=1)          # (B,L+1,D)
                return t + 1, dec_prefix, ta

            _, dec_prefix, ta = tf.while_loop(
                cond,
                body,
                loop_vars=[t0, dec_prefix, ta],
                parallel_iterations=1,
                shape_invariants=[
                    t0.get_shape(),
                    tf.TensorShape([None, None, self.feature_dim]),  # dec_prefix cresce in L
                    tf.TensorShape([]),                               # TensorArray
                ],
            )

            Y_pred = tf.transpose(ta.stack(), [1, 0, 2])  # (B,T_eff,D)
            Y_true = Y[:, :T_out, :]                      # (B,T_eff,D)

            # come nel LSTM: loss manuale (qui puoi usare la tua masked loss se vuoi)
            loss = self._masked_mse_loss(Y_true, Y_pred)

        grads_raw = tape.gradient(loss, self.trainable_variables)
        grads = cast(Sequence[Optional[tf.Tensor]], grads_raw)
        grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
        if not grads_and_vars:
            raise RuntimeError("There are no gradients")

        g_list, v_list = zip(*grads_and_vars)
        g_list, _ = tf.clip_by_global_norm(g_list, self.current_clip_norm)
        self.optimizer.apply_gradients(zip(g_list, v_list))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


    def test_step(self, data):
        X, Y = data
        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        # validation: sempre full t_out (come LSTM test_step)
        T_out = tf.cast(self.rt.t_out, tf.int32)

        memory = self._encode(X, training=False)
        dec_prefix = self._init_dec0(X)  # (B,1,D)

        ta = tf.TensorArray(
            dtype=Y.dtype,
            size=T_out,
            element_shape=tf.TensorShape([None, self.feature_dim]),
        )
        t0 = tf.constant(0, tf.int32)

        def cond(t, dec_prefix, ta):
            return t < T_out

        def body(t, dec_prefix, ta):
            y_seq = self._decode_prefix(dec_prefix, memory, training=False)
            y_t   = y_seq[:, -1:, :]                                # (B,1,D)
            ta = ta.write(t, tf.squeeze(y_t, axis=1))               # (B,D)

            # teacher forcing in validation (come nel tuo LSTM test_step)
            y_true_t = Y[:, t:t+1, :]                               # (B,1,D)
            dec_prefix = tf.concat([dec_prefix, y_true_t], axis=1)  # (B,L+1,D)
            return t + 1, dec_prefix, ta

        _, dec_prefix, ta = tf.while_loop(
            cond,
            body,
            loop_vars=[t0, dec_prefix, ta],
            parallel_iterations=1,
            shape_invariants=[
                t0.get_shape(),
                tf.TensorShape([None, None, self.feature_dim]),
                tf.TensorShape([]),
            ],
        )

        Y_pred = tf.transpose(ta.stack(), [1, 0, 2])      # (B,T_out,D)
        Y_true = Y[:, :T_out, :]                          # (B,T_out,D)

        # val_loss sul full T_out (così matcha “val_loss and fr_loss: 75”)
        loss = tf.reduce_mean(tf.square(Y_true - Y_pred))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
