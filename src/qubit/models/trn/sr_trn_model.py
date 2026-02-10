from __future__ import annotations

from ...enums.loss_on import LossOn

from ...utils.layers_names import ENC_BLOCK_, ENC_IN_DROP, ENC_IN_PROJ, ENC_POS_EMB, OUT_DENSE


from ...dataclasses.sr_config import SuperResolutionConfig
import tensorflow as tf
import keras
from keras import layers

from .encoder import EncoderTRNBlock

class SrTrnModel(keras.Model):

    def __init__(
        self,
        *,
        feature_dim: int, # output feature_dim (magnetizzations and correlations)
        input_feature_dim: int, # feature_dim +1 (mask channel)
        windows_len: int, # (high-res window length) == input_seq_len * sr.stride == ouput_seq_len
        dim_model: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
        sr_cfg: SuperResolutionConfig,
        name: str = "sr_trn_encoder",
    ):
        super().__init__(name=name)
        self.feature_dim = int(feature_dim)
        self.input_feature_dim = int(input_feature_dim)
        self.windows_len = int(windows_len)
        self.sr_cfg = sr_cfg
        self.dim_model = dim_model

        # Projections (continuous features -> dim_model)
        # needed because the internal rappresentations of transformer
        # work with dim_model as third dimension and not (feature_dim + 1)
        self.enc_in = layers.Dense(dim_model, name=ENC_IN_PROJ)

        # input  => pos_idx.shape(window_size,) => list of index from 0 to window_size - 1
        # output => pos_emb.shape(window_size, dim_model) => pos_emb[tf.newaxis, : ,: ].shape(1, window_size, dim_model)
        self.enc_pos = layers.Embedding(input_dim=windows_len, output_dim=dim_model, name=ENC_POS_EMB)

        # dropout layer don't change the shape so the input.shape == output.shape
        self.enc_drop = layers.Dropout(dropout, name=ENC_IN_DROP)

        self.enc_blocks = [
            EncoderTRNBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                name=f"{ENC_BLOCK_}{i}",
            )
            for i in range(num_layers)
        ]

        # input => (batch_size, window_size, d_model)
        # output => (batch_size, window_size, feature_dim)
        self.out_dense = layers.Dense(feature_dim, name=OUT_DENSE)

        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker] + super().metrics


    def build(self, input_shape: tf.TensorShape):
        
        # input_shape => X.shape(batch_size,window_size, input_feature_dim)
        
        # self.enc_in => layers.Dense 
        # input => X.shape(batch_size, window_size,input_feature_dim)
        # output => (batch_size, window_size, dim_model)
        self.enc_in.build(input_shape)

        # self.out_dense => layers.Dense 
        # input => y_pred.shape(batch_size, window_size,dim_model)
        # output => (batch_size, window_size, feature_dim)
        self.out_dense.build((None, None, self.dim_model))

        super().build(input_shape)

    def call(self, X: tf.Tensor, training: bool = False) -> tf.Tensor:
        # X.shape(batch_size, window_size, input_feature_dim == feauture_dim  + 1 (mask channel))
        
        windows_size = tf.shape(X)[1]
        # window_size
        
        # layers.Dense 
        # input => X.shape(batch_size , window_size, input_feature_dim)
        # output => x.shape(batch_size, window_size, dim_model)
        x = self.enc_in(X)  

        # pos => list of index from 0 to L-1
        pos = tf.range(windows_size, dtype=tf.int32)
        
        # layers.Embedding = pos.shape (windows_size,) => dec_pos.shape(windows_size,dim_model)
        # tf.nexaxis = input dec_pos.shape(windows_size,dim_model) => pos_emb.shape (1,windows_size,dim_model)
        pos_emb =  self.enc_pos(pos)[tf.newaxis, :, :]  

        # broadcast on the batch_size dimension
        x = x + pos_emb
        # x.shape(batch_size, window_size, dim_model)

        # layers.Dropout doesn't change the shape
        x = self.enc_drop(x, training=training)

        for blk in self.enc_blocks:
            x = blk(x, training=training,return_attns = False)

        # layers.Dense 
        # input => y.shape (batch_size,window_size,dim_model)
        # output => y_pred.shape (batch_size,window_size,feature_dim)
        y_pred = self.out_dense(x) 

        return y_pred

    def _make_sample_weight(self, X: tf.Tensor) -> tf.Tensor:
        # X.shape(batch_size, window_size, input_feature_dim == feauture_dim  + 1 (mask channel))
        
        # obs mask is last channel  (mask): 
        # obs.shape(batch_size, window_size, 1)

        # we extract the mask channel == obs from input X
        # obs == 1 when the specific timestep t is observed (not a hole) 
        # obs == 0 when the specific timestep t is a hole  (missed) 
        obs = X[:, :, -1:]

        # miss is the opposite of obs 
        # miss == 1 when the specific timestep t is a hole  (missed) 
        # miss == 0 when the specific timestep t is observed (not a hole) 
        miss = 1.0 - obs
        # miss.shape(batch_size, windows_size, 1)

        # if the loss is calculated only to missed values 
        if self.sr_cfg.loss_on == LossOn.MISSING: w = miss
        # instead the loss is calculated for all values but for the values observed we use a dowmnweight to stabilize the error
        else: w = miss + tf.cast(self.sr_cfg.observed_weight, X.dtype) * obs

        # squeeze
        # previous w.shape(batch_size, windows_size, 1)
        # after w.shape(batch_size, windows_size)
        return tf.squeeze(w, axis=-1)

    def train_step(self, data):

        X, Y_true = data  

        # ! input_feature_dim == feature_dim + 1 (channel mask)

        # X_train => X.shape(batch_size, windows_size, input_feature_dim)
        # Y_train => Y_true.shape(batch_size, windows_size, feature_dim)
        
        sw = self._make_sample_weight(X)
        # sw.shape(batch_size, windows_size)

        with tf.GradientTape() as tape:
            y_pred = self(X, training=True) # call the call function
            # y_pred.shape (batch_size,window_size,feature_dim)

            loss = self.compute_loss(X, Y_true, y_pred, sample_weight=sw)

        # calculate gradients (backward pass)
        grads = tape.gradient(loss, self.trainable_variables) # type: ignore[reportCallIssue]
        
        # backpropagation => update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)      
        # update metrics (includes the metric that tracks the loss)
        self.compute_metrics(x = X, y = Y_true, y_pred = y_pred, sample_weight=sw)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):

        X, Y_true = data  

        # ! input_feature_dim == feature_dim + 1 (channel mask)

        # X_val => X.shape(batch_size, windows_size, input_feature_dim)
        # Y_val => Y_true.shape(batch_size, windows_size, feature_dim)
        
        sw = self._make_sample_weight(X)
        # sw.shape(batch_size, windows_size)

        y_pred = self(X, training=False)
        val_loss = self.compute_loss(X, Y_true, y_pred, sample_weight=sw)

        self.loss_tracker.update_state(val_loss)
        self.compute_metrics(X, Y_true, y_pred, sample_weight=sw)

        return {m.name: m.result() for m in self.metrics}
