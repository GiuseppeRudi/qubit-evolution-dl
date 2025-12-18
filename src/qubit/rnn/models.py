import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed


def build_rnn_model(input_shape, output_seq_len, feature_dim):
    """
    Costruisce un modello Seq2Seq (Encoder-Decoder) basato su LSTM.
    Questo è l'approccio standard per la previsione di sequenze con RNN.

    Args:
        input_shape (tuple): (input_seq_len, feature_dim)
        output_seq_len (int): Lunghezza della sequenza di output da prevedere.
        feature_dim (int): Dimensione delle feature.

    Returns:
        tf.keras.Model: Il modello Seq2Seq compilato.
    """
    input_seq_len, _ = input_shape
    latent_dim = 64  # Dimensione dello spazio latente

    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape)
    # L'LSTM dell'Encoder elabora la sequenza di input e restituisce lo stato finale
    # (h e c) che cattura il "contesto" della sequenza di input.
    encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_inputs)
    encoder_states = [state_h, state_c]

    # --- Decoder ---
    # Il Decoder riceve lo stato finale dell'Encoder come stato iniziale.
    # Per la previsione di sequenze, usiamo un approccio che decodifica
    # l'intero output in una volta sola (più semplice per Keras).

    # 1. Ripetiamo il vettore di contesto (state_h) per la lunghezza della sequenza di output.
    # Questo fornisce al decoder il contesto ad ogni passo temporale.
    decoder_input = RepeatVector(output_seq_len)(state_h)

    # 2. LSTM del Decoder. Non usiamo return_sequences=False perché vogliamo l'output per ogni passo.
    decoder_lstm = LSTM(latent_dim, return_sequences=True)(decoder_input, initial_state=encoder_states)

    # 3. TimeDistributed Dense Layer: Applica lo stesso Dense Layer ad ogni passo temporale
    # per mappare l'output dell'LSTM alla dimensione delle feature (feature_dim).
    decoder_outputs = TimeDistributed(Dense(feature_dim))(decoder_lstm)

    # --- Modello ---
    model = Model(encoder_inputs, decoder_outputs, name="RNN_Seq2Seq_Model")

    # Compilazione: usiamo MSE (Mean Squared Error) che è comune per la regressione di sequenze
    model.compile(optimizer='adam', loss='mse')

    return model


if __name__ == '__main__':
    # Parametri aggiornati (corrispondenti al dataset trajectories.csv)
    INPUT_SEQ_LEN = 100
    OUTPUT_SEQ_LEN = 901  # 1001 - 100 = 901
    FEATURE_DIM = 55  # 10 Magnetizzazioni + 45 Correlazioni

    input_shape = (INPUT_SEQ_LEN, FEATURE_DIM)

    rnn_model = build_rnn_model(input_shape, OUTPUT_SEQ_LEN, FEATURE_DIM)
    rnn_model.summary()

    print("\nModello RNN (LSTM Seq2Seq) creato con successo.")
