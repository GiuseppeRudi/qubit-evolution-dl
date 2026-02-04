from typing import Optional, Tuple
import tensorflow as tf
import keras
from keras import layers

from ...utils.layers_names import _LN1, _LN2, _DROP1, _DROP2, _MHA, _FFN

class EncoderTRNBlock(layers.Layer):
    def __init__(self, *, dim_model: int, num_heads: int, ff_dim: int, dropout: float, name: str):
        super().__init__(name=name)

        self.last_attn_scores: tf.Tensor | None = None

        # normalize each element v(batch_size, input_seq_len, :) => v.shape(dim_model,)
        # each timestep of each element in the batch is normalized independently
        # doesn't change the input shape so input shape == output shape 
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}{_LN1}")
        
        # MULTI HEAD ATTENTION => SELF
        # internally we work with 3 linear transformation : Q , K , V

        # each transformation work with a number of heads and each head have a dim_head dimension
        # dim_head = key_dim = dim_model // num_heads

        # Q.shape(batch_size, num_heads, input_seq_len, dim_head)
        # K.shape(batch_size, num_heads, input_seq_len, dim_head)
        # V.shape(batch_size, num_heads, input_seq_len, dim_head)

        # Conceptually:
        # Self-attention (encoder):
        # For each timestep, each "head" decides which other timesteps to pay attention to.
        # The query is "what do I need", the keys are "what information do I have", and the values are "the content".
        # Comparing Query with all Keys produces weights: the more relevant another timestep is, the more it is weighted.
        
        # We apply the dropout on the attentions map because we want to tell at the model 
        # to not always rely on the same connections between timesteps, so some connections are turned off
        
        # The output is a weighted combination of the Values of the other timesteps.
        # (in the encoder, it can look at the entire sequence, so no causal mask)
  
        # Practically 
        # for each head and for each timesteps t, compare Q[t] with all K[s]
        # computing the scores with a scaled dot product (batch_size, num_heads, input_seq_len, input_seq_len)
        
        # after we apply the softmax on the last dimension and we obtain the attention weights 
        # weights : (batch_size, num_heads, input_seq_len, input_seq_len)
        # each row adds up to 1: “as much as timestep t looks at each timestep s”

        # apply the dropout  

        # we compute the weighted mean and concatenate all heads 
        # context per head (batch_size, num_heads, input_seq_len, d_head)
        # concat : (batch_size, input_seq_len, num_heads * d_head) = (batch_size, input_seq_len, dim_model)
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim_model // num_heads,
            dropout=dropout,
            name=f"{name}{_MHA}",
        )

        # we use a layer of Dropout on the ouput attention to regularize the contribution
        # of the attention to the residual streams 
        self.drop1 = layers.Dropout(dropout, name=f"{name}{_DROP1}")

        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}{_LN2}")
        
        
        self.ffn = keras.Sequential(
            [   
                # x.shape(batch_size, input_seq_len, dim_model) => x.shape(batch_size, input_seq_len, ff_dim)
                # relu: adds non-linearity so the FFN can learn more complex transformations.
                layers.Dense(ff_dim, activation="relu"),
                
                layers.Dropout(dropout),

                # x.shape(batch_size, input_seq_len, ff_dim) => x.shape(batch_size, input_seq_len, dim_model)
                layers.Dense(dim_model),
            ],
            name=f"{name}{_FFN}",
        )
        self.drop2 = layers.Dropout(dropout, name=f"{name}{_DROP2}")

    def call(self, x: tf.Tensor, *, training: bool, return_attns : bool) -> tf.Tensor:
        # x is already well structured => (dense, positional embedding, pre-dropout)
        # x.shape(batch_size, input_seq_len, dim_model)
        
        # ! token is a general term to indicate a piece of sequence 
        # in our case we work with the temporal sequence so token == timestep

        # PreNorm self-attn
        y = self.ln1(x)
        # y.shape(batch_size, input_seq_len, dim_model)
        
        # since query = key = value => self-attention 
        # each timestep (each piece of sequence) can look all other timestep of the same sequence 

        if return_attns:
            attn_out, attn_scores = self.mha(
                query=y, value=y, key=y,
                return_attention_scores=True,
                training=training,
            )
            self.last_attn_scores = attn_scores   
            # attn_scores(batch_size, num_heads, input_seq_len, input_seq_len)

        else:
            attn_out = self.mha(query=y, value=y, key=y, training=training)
            self.last_attn_scores = None
        
        # attn_out.shape(batch_size, input_seq_len, dim_model)
        # attn_scores(batch_size, num_heads, input_seq_len, input_seq_len)

        # Residual Stream
        
        # x (new representation) =  x (current representation) + attn (update)
        # x (batch_size, input_seq_len, dim_model) = x (batch_size,input_len,dim_model) + attn(batch_size, input_seq_len, dim_model)
        x = x + self.drop1(attn_out, training=training)

        # PreNorm FFN
        y = self.ln2(x)
        # y.shape(batch_size, input_seq_len, dim_model)

        f = self.ffn(y, training=training)
        # f.shape(batch_size, input_seq_len, dim_model)

        # x (new representation) =  x (current representation) + f (update)
        # x (batch_size, input_seq_len, dim_model) = x(batch_size,input_len,dim_model) + f(batch_size, input_seq_len, dim_model)
        x = x + self.drop2(f, training=training)
        
        return x

