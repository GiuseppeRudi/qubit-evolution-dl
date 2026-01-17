# import tensorflow as tf

# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import (
#     Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
#     Embedding, GlobalAveragePooling1D, Layer, Reshape
# )


# # --- 1. Positional Encoding (Cruciale per i Transformer) ---
# class PositionalEmbedding(Layer):
#     def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
#         super().__init__(**kwargs)
#         # L'Embedding layer non è strettamente necessario per i dati numerici di serie temporali,
#         # ma è incluso per completezza se i dati fossero discreti.
#         # Per i dati numerici, useremo direttamente l'input e aggiungeremo solo l'encoding posizionale.
#         self.sequence_length = sequence_length
#         self.embed_dim = embed_dim

#         # Inizializza l'encoding posizionale (vettori di posizione)
#         self.position_embeddings = Embedding(
#             input_dim=sequence_length, output_dim=embed_dim
#         )

#     def call(self, inputs):
#         # inputs shape: (batch_size, sequence_length, feature_dim)

#         # Crea gli indici di posizione: [0, 1, 2, ..., sequence_length-1]
#         positions = tf.range(start=0, limit=self.sequence_length, delta=1)

#         # Calcola l'embedding posizionale
#         embedded_positions = self.position_embeddings(positions)

#         # Aggiunge l'embedding posizionale all'input (assumendo che l'input sia già mappato a embed_dim)
#         # Per semplicità, assumiamo che l'input sia già mappato alla dimensione dell'embedding (embed_dim)
#         # Se l'input non è mappato, l'aggiunta diretta non è corretta.
#         # Per i dati di serie temporali, mappiamo prima l'input a embed_dim.

#         # Mappiamo l'input alla dimensione dell'embedding (se feature_dim != embed_dim)
#         # In questo esempio, assumiamo che l'input sia già proiettato o che feature_dim = embed_dim.

#         return inputs + embedded_positions


# # --- 2. Transformer Block (Encoder Layer) ---
# class TransformerBlock(Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = Sequential(
#             [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
#         )
#         self.layernorm1 = LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = LayerNormalization(epsilon=1e-6)
#         self.dropout1 = Dropout(rate)
#         self.dropout2 = Dropout(rate)
#         self.embed_dim = embed_dim

#     def call(self, inputs, training=False):
#         # Multi-Head Attention
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         # Add & Norm
#         out1 = self.layernorm1(inputs + attn_output)

#         # Feed Forward
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         # Add & Norm
#         return self.layernorm2(out1 + ffn_output)
