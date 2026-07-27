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

class HybridTrnModel(StrategyChooserModel):
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
        t_in: int,
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

        super().build(input_shape)

    def _causal_mask(self, t: tf.Tensor) -> tf.Tensor:
        
        # create matrix of dimension t*t with all values = 1 
        ones = tf.ones((t, t), dtype=tf.float32)
        # one.shape(t,t)
        
        # below the main diagonal and the main diagonal itself => values = 1 
        # above the main diagonal => values = 0  
        # band.shape(t,t) 
        band = tf.linalg.band_part(ones, -1, 0) 

        # where the values of band > 0 set True
        # otherwise set False
        # mask.shape(t,t)
        mask = tf.convert_to_tensor(band > 0.0, tf.bool)

        # add the third dimensions for broadcasting on batch_size
        # mask.shape(1,t,t)
        return mask[tf.newaxis, :, :] 

    def _encode(self, X: tf.Tensor, *, training: bool, return_attns : bool = False ) -> tf.Tensor:
        
        # X.shape(batch_size, input_seq_len, feature_dim)
        # X: (batch_size,input_seq_len,feature_dim) -> memory: (batch_size,input_seq_len,dim_model)

        input_seq_len = tf.shape(X)[1]

        # layer.Dense = (:,:,feature_dim) => (:,:,dim_model)
        x = self.enc_in(X)

        # list of indexes from 0 to input_seq_len - 1
        pos = tf.range(input_seq_len, dtype=tf.int32)

        # layers.Embedding => pos.shape(input_seq_len,) => (input_seq_len,dim_model)
        # tf.nexaxis => input (input_seq_len,dim_model) => (1,input_seq_len, dim_model)
        pos_emb = self.enc_pos(pos)[tf.newaxis, :, :]
        
        # broadcast on batch_size
        # x (batch_size, input_seq_len, dim_model ) = x (batch_size, input_seq_len, dim_model ) + pos_emb(1, input_seq_len, dim_model)
        x = x + pos_emb

        # layers.Dropout 
        # doesn't change the shape of x 
        x = self.enc_drop(x, training=training)

        # at the beginning, x is the projected input + positional + dropout
        # when an encoder block return x, the next will use the returned x to update it in a more refined way
        for blk in self.enc_blocks: 
            x = blk(x, training=training, return_attns = return_attns)

        x = tf.ensure_shape(x, [None, None, self.dim_model])
        return x

    def _init_dec0(self, X: tf.Tensor) -> tf.Tensor:

        # Inizialize decoder input at the t (timestep) = 0 
        # X.shape(batch_size , input_seq_len, feature_dim)

        #  ! Initialization of subsequent timesteps occurs when entering strategy functions.

        if self.start_mode == StartMode.LAST_X:
            input_seq_len = tf.shape(X)[1]
            # take the last timestep 

            last = tf.gather(X, input_seq_len - 1, axis=1)
            # last.shape(batch_size, feature_dim)
            
            # add the timestep dimension 
            return tf.expand_dims(last, axis=1)  # (batch_size, 1, feature_dim)

        batch_size = tf.shape(X)[0]
        feature_dim = self.feature_dim
        
        # returns an array with shape(batch_size, 1 , feature_dim) all filled with zeros
        return tf.zeros(tf.stack([batch_size, 1, feature_dim]), dtype=X.dtype)

    def _decode_prefix(self, dec_prefix: tf.Tensor, memory: tf.Tensor, *, training: bool, return_attn : bool = False) -> tf.Tensor:
       
        # encoder ouput => memory.shape(batch_size, input_seq_len, dim_model)

        # L =  t (current timesteps) + 1
        L = tf.shape(dec_prefix)[1]

        # dec_prefix is the decoder input sequence, shape (batch_size, L, feature_dim)
        # where L depends on the decoding mode:
        # - FULL_SEQ: L = T_out (or T_hor)  -> fixed length
        # - STEP_WISE (prefix growing): at loop step t (0-based), L = t+1 -> grows 1..T_out (or T_hor)

        # if decoder_mode_id == STEP_WISE (index: 1)
        # if t=0 dec_prefix.shape(batch_size,1, feature_dim) where L = t+1 = 1, with values (LAST_X or ZEROS) based on StartMode
        # if t>0 dec_prefix.shape(batch_size,L, feature_dim) 
        
        # layers.Dense 
        # input  => dec_prefix.shape (batch_size,L,feauture_dim)
        # output => y.shape (batch_size,L,dim_model)
        y = self.dec_in(dec_prefix) 
        
        # pos => list of index from 0 to L-1
        pos = tf.range(L, dtype=tf.int32)

        # layers.Embedding = pos.shape (L,) => dec_pos.shape(L,dim_model)
        # tf.nexaxis = input dec_pos.shape(L,dim_model) => pos_emb.shape (1,L,dim_model)
        pos_emb = self.dec_pos(pos)[tf.newaxis, :, :]  
        
        # broadcast on the batch_size dimension
        y = y + pos_emb
        # y.shape(batch_size, L, dim_model)
        
        # layers.Dropout doesn't change the shape
        y = self.dec_drop(y, training=training)

        causal = self._causal_mask(L)  
        # causal.shape(1,L,L)

        # when an decoder block return y, the next will use the returned y to update it in a more refined way
        for blk in self.dec_blocks:
            y = blk(y, memory=memory, causal_mask=causal, training=training, return_attn = return_attn)

        # y.shape(batch_size, L, dim_model)

        # layers.Dense 
        # input => y.shape (batch_size,L,dim_model)
        # output => y_pred.shape (batch_size,L,feature_dim)
        y_pred = self.out_dense(y)  

        return y_pred

    def forward_with_attn(self, X: tf.Tensor, Y: tf.Tensor, *, training: bool = False):
        
        # ! important to compute the attention maps we use the teacher forcing
        
        # use 1 sample 
        # X : (batch_size = 1, input_seq_len, feature_dim)
        # Y : (batch_size = 1, output_seq_len, feature_dim)
        attn_maps: dict[str, tf.Tensor] = {}

        memory = self._encode(X, training=training, return_attns=True)
        dec0 = self._init_dec0(X)

        self.rt.phase_id.assign(0) # teacher forcing
        dec_in = self.apply_strategy_full_seq(Y, dec0=dec0)

        y_pred = self._decode_prefix(dec_in, memory, training=training, return_attn=True)

        for i, blk in enumerate(self.enc_blocks):
            if blk.last_attn_scores is not None:
                attn_maps[f"enc/l{i}/self"] = blk.last_attn_scores

        for i, blk in enumerate(self.dec_blocks):
            if blk.last_self_scores is not None:
                attn_maps[f"dec/l{i}/self"] = blk.last_self_scores
            if blk.last_cross_scores is not None:
                attn_maps[f"dec/l{i}/cross"] = blk.last_cross_scores

        return y_pred, attn_maps
    

    @property
    def metrics(self):
        return [self.loss_tracker] + super().metrics

    def train_step(self, data):
        X, Y = data
        
        # X.shape(batch_size, input_seq_len, feature_dim)
        # Y.shape(batch_size, output_seq_len, feature_dim)
        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        decoder_mode_id = tf.convert_to_tensor(self.rt.decoder_mode_id) # 0 => FULL_SEQ | 1 => STEP_WISE
        prediction_mode_id = tf.convert_to_tensor(self.rt.prediction_mode_id)  # 0 => ALL | 1 => HORIZON
        T_out = tf.convert_to_tensor(self.rt.t_out)
        T_hor = tf.convert_to_tensor(self.rt.horizon)
        
        # IF  PREDICTION_MODE == ALL  (index 0) while condition is t < T_out
        # IF  PREDICTION_MODE == HOR  (index 1) while condition is t < T_hor
        T = tf.cond(
            tf.equal(prediction_mode_id, 0),
            lambda: T_out,
            lambda: T_hor,
        )

        def run_full_seq():

            def run_all():
                
                # decoder input  dec_in.shape(batch_size , T_out == output_seq_len, feature_dim)
                dec_in = self.apply_strategy_full_seq(Y, dec0=dec_prefix)

                # Y_pred.shape(batch_size, T_out == output_seq_len , feature_dim)
                Y_pred = self._decode_prefix(dec_in, memory, training=True)
                
                # so we predict to all T_out to stabilize the graph and not occur the tracing
                # but we return only the part we need (horizon)
                return Y[:, :T_hor, :], Y_pred[:, :T_hor, :]

            def run_hor_only():

                # Y.shape(batch_size, T_out, feature_dim)
                # Y_hor(batch_size , T_hor, feature_dim)
                Y_hor = Y[:, :T_hor, :]

                dec_in = self.apply_strategy_full_seq(Y_hor, dec0=dec_prefix) 
                # decoder_input dec_in.shape(batch_size, T_hor, feature_dim)  

                # Y_pred.shape(batch_size, T_hor , feature_dim)
                Y_pred = self._decode_prefix(dec_in, memory, training=True)
                                    
                return Y_hor, Y_pred

            Y_true, Y_pred = tf.cond(tf.equal(prediction_mode_id, 0), run_all, run_hor_only)

            loss = self.compute_loss(y=Y_true, y_pred=Y_pred)
            return loss, Y_true, Y_pred


        def run_step_wise():

            # array with T elements and each element have shape(batch_size, feature_dim)
            # batch_size is None because is dinamic 
            ta = tf.TensorArray(
                dtype=Y.dtype,
                size=T,
                element_shape=tf.TensorShape([None, self.feature_dim]),
            )

            # ta = shape (batch_size, T, feature_dim)
            # print(ta.element_shape) = shape (batch_size, feature_dim) ,  ta.size() = T_out

            t0 = tf.constant(0, tf.int32)

            def cond(t, dec_prefix, ta):
                return t < T

            def body(t, dec_prefix, ta):
                # if t = 0  dec_prefix.shape(batch_size, 1, feature_dim)
                # if t > 0  dec_prefix.shape(batch_size, t+1, feature_dim)

                # encoder_ouput => memory.shape(batch_size, 1, dim_model)

            
                y_pred_seq = self._decode_prefix(dec_prefix, memory, training=True)  # (batch_size,t+1,feature_dim)
                y_pred_t = y_pred_seq[:, -1:, :] # (batch_size,1,feature_dim)
                
                y_pred_t_2d = tf.squeeze(y_pred_t, axis=1)
                # y_pred_t_2d.shape(batch_size, feature_dim)

                ta = ta.write(t, y_pred_t_2d)
                # ta.shape(index = t , element.shape(batch_size, feature_dim))

                # Ground truth Y.shape(batch_size, timesteps, feature_dim)

                y_true_2d = tf.gather(Y, t, axis=1)             
                # y_true_2d.shape(batch_size, feature_dim)
                
                # where index = t in y_true_2d and y_pred_t_2d
                y_true_t = tf.expand_dims(y_true_2d, axis=1)
                # y_true_t.shape(batch_size, 1, feature_dim)

                next_in = self.apply_strategy_step_wise(
                    y_true_t = y_true_t,
                    y_pred_t = y_pred_t,
                )

                # choose as send the current prediction based on the strategy to the decoder input of the next timestep      
                # next_in.shape(batch_size,1, feature_dim)
                
                # previous => dec_prefix.shape(batch_size, t+1 == L, feature_dim)
                dec_prefix = tf.concat([dec_prefix, next_in], axis=1) 
                # after =>    dec_prefix.shape(batch_size,(t+1)+1 == L+1,feature_dim)         
                return t + 1, dec_prefix, ta


            _, _, ta = tf.while_loop(
                cond,
                body,
                loop_vars=[t0, dec_prefix, ta],
                parallel_iterations=1,
                shape_invariants=[
                    t0.get_shape(),
                    # L changes, so without shape_invariants the loop fail
                    tf.TensorShape([None, None, self.feature_dim]),
                    tf.TensorShape([]),
                ],
            )

            # tf.print(ta.stack())
            # stack: (element_size = T, batch_size, feature_dim) -> transpose: (batch_size, T, feature_dim)
            Y_pred = tf.transpose(ta.stack(), [1, 0, 2])

            # Y_true: (N, T, D)
            Y_true = tf.slice(Y, [0, 0, 0], [-1, T_hor, -1])

            Y_pred = tf.cond(
                tf.equal(prediction_mode_id, 0),
                lambda: tf.slice(Y_pred, [0, 0, 0], [-1, T_hor, -1]),
                lambda: Y_pred,
            )
    
            loss = self.compute_loss(y=Y_true, y_pred=Y_pred)
            return loss, Y_true, Y_pred

        with tf.GradientTape() as tape:

            # self._encode is run once for both decoder mode (STEP_WISE and FULL_SEQ)
            memory = self._encode(X, training=True)  # (batch_size,input_seq_len,dim_model)
            dec_prefix = self._init_dec0(X)  # (batch_size,1,feature_dim)
    
            loss, Y_true, Y_pred = tf.switch_case(
                decoder_mode_id,  
                branch_fns={
                    0: run_full_seq,
                    1: run_step_wise,
                },
            )

        # calculate gradients (backward pass)
        grads = tape.gradient(loss, self.trainable_variables) # type: ignore[reportCallIssue]

  
        # filter the None gradients => take the variables that have gradients
        grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
        if not grads_and_vars:
            raise RuntimeError("There are no gradients")
        
        # gradient clipping => prevents gradient explosions by limiting the global norm
        g_list, v_list = zip(*grads_and_vars)
        g_list, _ = tf.clip_by_global_norm(g_list, self.current_clip_norm)
        
        # backpropagation => update weights
        self.optimizer.apply_gradients(zip(g_list, v_list))

        self.loss_tracker.update_state(loss)      
        # update metrics (includes the metric that tracks the loss)
        self.compute_metrics(x = X, y = Y_true, y_pred = Y_pred)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        X, Y = data

        # X_val => X.shape => (batch_size, input_seq_len, feature_dim )
        # Y_val => Y.shape => (batch_size, output_seq_len, feature_dim )
        
        X = tf.ensure_shape(X, [None, None, self.feature_dim])
        Y = tf.ensure_shape(Y, [None, None, self.feature_dim])

        # ! dec_in is always the ground truth => teacher forcing + dec0
        # ! Y_pred second dimension is always => horizon => coerent with train_step
       
         # self.rt.horizon when is = -1 so self.rt.horizon == self.rt.t_out
        T_hor = tf.convert_to_tensor(self.rt.horizon)

        # inititialize to zeros or last_x based on the start_mode parameter
        dec0 = self._init_dec0(X)         

        prediction_mode_id = tf.convert_to_tensor(self.rt.prediction_mode_id)  # 0 => ALL | 1 => HORIZON
        
        # dec_in(batch_size,  t = T_out || T_hor, feature_dim )  
        
        # ! in full_seq  dec_in is the prediction help that contains all the information
        # ! that the decoder must predict using ground truth (with the possibility of masking) 

        memory = self._encode(X, training=False)
        # memory.shape(batch_size,input_seq_len,dim_model)

        def run_all():
            
            # decoder input  dec_in.shape(batch_size , T_out == output_seq_len, feature_dim)
            dec_in = self.apply_strategy_full_seq(Y, dec0=dec0)

            # Y_pred.shape(batch_size, T_out == output_seq_len , feature_dim)
            Y_pred = self._decode_prefix(dec_in, memory, training=True)
            
            # so we predict to all T_out to stabilize the graph and not occur the tracing
            # but we return only the part we need (horizon)
            return Y[:, :T_hor, :], Y_pred[:, :T_hor, :]

        def run_hor_only():

            # Y.shape(batch_size, T_out, feature_dim)
            # Y_hor(batch_size , T_hor, feature_dim)
            Y_hor = Y[:, :T_hor, :]

            dec_in = self.apply_strategy_full_seq(Y_hor, dec0=dec0) 
            # decoder_input dec_in.shape(batch_size, T_hor, feature_dim)  

            # Y_pred.shape(batch_size, T_hor , feature_dim)
            Y_pred = self._decode_prefix(dec_in, memory, training=True)
                                
            return Y_hor, Y_pred
        
        self.rt.phase_id.assign(0) # teacher forcing strategy

        Y_true, Y_pred = tf.cond(tf.equal(prediction_mode_id, 0), run_all, run_hor_only)
        
        val_loss = self.compute_loss(y=Y_true, y_pred=Y_pred)
        self.loss_tracker.update_state(val_loss)

        self.compute_metrics(x=X, y=Y_true, y_pred=Y_pred)

        return {m.name: m.result() for m in self.metrics}
