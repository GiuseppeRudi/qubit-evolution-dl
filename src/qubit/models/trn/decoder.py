import tensorflow as tf
import keras
from keras import layers

from ...utils.layers_names import _LN1, _LN2, _LN3, _DROP1, _DROP2, _DROP3, _SELF_MHA, _CROSS_MHA, _FFN

class DecoderTRNBlock(layers.Layer):
    def __init__(self, *, dim_model: int, num_heads: int, ff_dim: int, dropout: float, name: str):
        super().__init__(name=name)

        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}{_LN1}")
        
        # Decoder self-attention with causal mask: each timestep can attend only to past/current positions (no future look-ahead).
        self.self_mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim_model // num_heads,
            dropout=dropout,
            name=f"{name}{_SELF_MHA}",
        )
        self.drop1 = layers.Dropout(dropout, name=f"{name}{_DROP1}")

        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}{_LN2}")
        
        
        # MULTI HEAD ATTENTION => CROSS
        # internally we work with 3 linear transformation : Q , K , V

        # each transformation work with a number of heads and each head have a dim_head dimension
        # dim_head = key_dim = dim_model // num_heads

        # Q.shape(batch_size, num_heads, T, dim_head)
        # K.shape(batch_size, num_heads, T, dim_head)
        # V.shape(batch_size, num_heads, T, dim_head)

        # Conceptually:
        # Cross-attention (decoder -> encoder):
        # For each decoder timestep, each "head" decides which encoder timesteps (the input/memory) to pay attention to.
        # The query is "what do I need now in the decoder", while keys/values come from the encoder memory:
        # keys are "what information is available in the input", and values are "the input content to retrieve".
        # Comparing decoder Queries with encoder Keys produces weights: the more relevant an encoder timestep is,
        # the more it is weighted. The output is a weighted combination of encoder Values.
        # (no causal mask here, because the encoder memory is fully available)

        # Dropout inside cross-attention helps the model not always rely on the same encoder-to-decoder connections,
        # by randomly turning off some attention links during training.

        # Practically:
        # for each head and for each decoder timestep t_dec, compare Q_dec[t_dec] with all K_enc[s_enc]
        # scores: (batch_size, num_heads, dec_len, enc_len)

        # after softmax on the last dimension we obtain attention weights:
        # weights: (batch_size, num_heads, dec_len, enc_len)
        # each row adds up to 1: "how much decoder timestep t_dec looks at each encoder timestep s_enc"

        # we compute the weighted sum of encoder Values and concatenate all heads:
        # context per head: (batch_size, num_heads, dec_len, d_head)
        # concat: (batch_size, dec_len, num_heads * d_head) = (batch_size, dec_len, dim_model)

        self.cross_mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim_model // num_heads,
            dropout=dropout,
            name=f"{name}{_CROSS_MHA}",
        )
        self.drop2 = layers.Dropout(dropout, name=f"{name}{_DROP2}")

        self.ln3 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}{_LN3}")
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(dim_model),
            ],
            name=f"{name}{_FFN}",
        )
        self.drop3 = layers.Dropout(dropout, name=f"{name}{_DROP3}")

    def call(
        self,
        x: tf.Tensor,
        *,
        memory: tf.Tensor,
        causal_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        
        # - FULL_SEQ: T = output_seq_len (t_out)
        # - STEP_WISE (prefix growing, no KV cache):
        #   T starts from 1 and increases each step (1..t_out).
        #   At each iteration we pass the whole decoded prefix to the decoder
        #   and take the last timestep output as the current prediction.
            
        # previous predictions => x.shape(batch_size, T, d_model) where:

        # output of encoder block => memory.shape(batch_size, input_seq_len, d_model)
        
        # causal_mask (1, T, T) broadcastable to (B, T, T):
        # prevents the decoder from attending to "future" timesteps (only past + current allowed).
        # FULL_SEQ: required, because the decoder input contains the whole sequence, so without the mask it would be cheating.
        # STEP_WISE (prefix growing): not highly necessary since the prefix has no future tokens, but we keep it for consistency 
        print(x.shape)
        # masked self-attn (causal)
        y = self.ln1(x)
        # y.shape(batch_size, T, dim_model)
        
        # since query = key = value => self-attention 
        # each timestep (each piece of sequence) can look only past + current timesteps on the same sequence 
        attn1 = self.self_mha(query=y, value=y, key=y, attention_mask=causal_mask, training=training)
        # attn1.shape(batch_size, T, dim_model)

        # Residual connection:
        # attn1 contains information aggregated from past+current decoder positions (due to the causal mask).
        # We add it back to the current representation to preserve the original signal and improve gradient flow.
        
        # x (new representation) =  x (current representation) + attn1(update => from decoder history)
        # x (batch_size, T, dim_model) = x (batch_size,T,dim_model) + attn1(batch_size, T, dim_model)
        x = x + self.drop1(attn1, training=training)

        y = self.ln2(x)
        # y.shape(batch_size, T, dim_model)
        
        # cross-attn (decoder queries -> encoder memory)
        # cross because the decoder looks at the encoder so it crosses two different sequences 

        attn2 = self.cross_mha(query=y, value=memory, key=memory, training=training)
        # attn2.shape(batch_size, T, dim_model)

        # x (new representation) =  x (current representation) + attn2(update => from encoder memory)
        # x (batch_size, T, dim_model) = x (batch_size,T,dim_model) + attn2(batch_size, T, dim_model)
        x = x + self.drop2(attn2, training=training)

        # FFN
        y = self.ln3(x)
        # y.shape(batch_size, T , dim_model)

        f = self.ffn(y, training=training)
        # f.shape(batch_size, t, dim_model)

        # x (new representation) =  x (current representation) + f (update)
        # x (batch_size, T, dim_model) = x(batch_size,T,dim_model) + f(batch_size, T, dim_model)
        x = x + self.drop3(f, training=training)
        return x
