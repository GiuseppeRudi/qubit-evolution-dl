import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from pathlib import Path

from numpy import random

from src.qubit.core.data import load_and_process_data
from src.qubit.rnn.models import build_rnn_model
from src.qubit.transformer.models import build_transformer_model  
from src.qubit.core.seed import set_seed
from src.qubit.core.utils import get_device




# --- Parametri di configurazione (corrispondenti al dataset reale) ---
INPUT_SEQ_LEN = 100
OUTPUT_SEQ_LEN = 901 # 1001 - 100 = 901
FEATURE_DIM = 55 # 10 Magnetizzazioni + 45 Correlazioni
EPOCHS = 100 # Aumentato per un addestramento più significativo
BATCH_SIZE = 32

def load_data(file_path=None, traj_fraction=0.025):
    """
    Carica (o genera) i dati per l'addestramento.

    - Se file_path è None, usa <project_root>/data/trajectories.csv
    - Se il file non esiste, prova comunque a chiamare load_and_process_data()
      (che può avere il suo default o può fallire e tornare None).
    """
    # main.py è nella root progetto -> parent è il project root
    project_root = Path(__file__).resolve().parent

    if file_path is None:
        file_path = project_root / "data" / "trajectories.csv"
    else:
        file_path = Path(file_path).expanduser().resolve()

    if not file_path.exists():
        print(f"⚠️ File non trovato: {file_path}")
        print("↪️ Provo a generare i dati chiamando load_and_process_data()...")

    # Passo esplicitamente il path, così non dipendo da default strani tipo ../../../...
    X, Y = load_and_process_data(file_path=str(file_path), traj_fraction=traj_fraction)

    if X is None or Y is None:
        print("❌ Impossibile caricare/generare i dati.")
        return None, None

    return X, Y

def train_and_evaluate(model, X_train, Y_train, model_name):
    """Addestra e valuta un modello."""
    print(f"\n--- Addestramento del Modello: {model_name} ---")
    
    # Addestramento simulato
    history = model.fit(
        X_train, 
        Y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.1
    )
    
    print(f"Addestramento completato. Perdita finale (MSE): {history.history['loss'][-1]:.4f}")
    
    # Simulazione della previsione
    sample_input = X_train[0:1]
    print(f"Sample Input: {sample_input.tolist()}")
    prediction = model.predict(sample_input)
    
    print(f"Previsione simulata per un campione di input:")
    print(f"  Forma dell'Input (Breve Termine): {sample_input.shape}")
    print(f"  Forma della Previsione (Lungo Termine): {prediction.shape}")
    print(f"  Previsione (primi 5 passi temporali):")
    np.set_printoptions(suppress=True, precision=16)
    print(prediction[0, :5, :])
    
    return history, prediction


def generate_all_plots(X_test, Y_test, transformer_prediction, rnn_prediction, sample_index=0, feature_index=0):
    """
    Genera un grafico per confrontare la dinamica reale con le previsioni dei modelli.
    """

    # 1. Estrai la feature per il campione selezionato
    # Input (Breve Termine)
    input_sequence = X_test[sample_index, :, feature_index]
    # Output Reale (Lungo Termine)
    true_output = Y_test[sample_index, :, feature_index]
    # Previsioni
    rnn_output = rnn_prediction[0, :, feature_index]
    transformer_output = transformer_prediction[0, :, feature_index]

    # 2. Combina Input e Output Reale per la traiettoria completa
    full_true_trajectory = np.concatenate([input_sequence, true_output])

    # 3. Crea l'asse temporale
    input_len = len(input_sequence)
    full_len = len(full_true_trajectory)

    time_axis = np.arange(full_len)

    # 4. Crea il grafico
    plt.figure(figsize=(15, 6))

    # Traiettoria Reale (Ground Truth)
    plt.plot(time_axis, full_true_trajectory, label='Dinamica Reale (Ground Truth)', color='black', linewidth=2)

    # Previsione RNN
    rnn_time_axis = np.arange(input_len, full_len)
    plt.plot(rnn_time_axis, rnn_output, label='Previsione RNN (LSTM)', color='red', linestyle='--')

    # Previsione Transformer
    transformer_time_axis = np.arange(input_len, full_len)
    plt.plot(transformer_time_axis, transformer_output, label='Previsione Transformer', color='blue', linestyle='--')

    # Linea di separazione Input/Output
    plt.axvline(x=input_len - 1, color='gray', linestyle=':', label='Fine Input (T=2.0)')

    # Etichette e Titolo

    # Determina il nome della feature
    if feature_index < 10:
        feature_name = f"Magnetizzazione m_{feature_index + 1}"
    else:
        # Correlazioni c_ij. La logica per i nomi c_ij è complessa, usiamo un nome generico
        feature_name = f"Correlazione c_{feature_index - 9}"

    plt.title(f"Confronto Previsione Dinamica Quantistica: {feature_name}")
    plt.xlabel("Passi Temporali (Input: 0-99, Output: 100-1000)")
    plt.ylabel("Valore della Feature")
    plt.legend()
    plt.grid(True)

    # Salva il grafico nella cartella predictions
    plot_path = os.path.join('predictions',
                             f"feature_{feature_index + 1}_{feature_name.replace(' ', '_').replace('$', '').replace('{', '').replace('}', '')}.png")
    plt.savefig(plot_path)
    plt.close()  # Chiude la figura per liberare memoria
    print(f"Grafico salvato: {plot_path}")
    return plot_path

def main():

    device = get_device()
    set_seed(42, deterministic=True)


    # 1. Caricamento dei dati
    X, Y = load_data()
    if X is None:
        return

    print(tf.sysconfig.get_build_info())
    print("GPU available:", tf.config.list_physical_devices('GPU'))

    input_shape = (INPUT_SEQ_LEN, FEATURE_DIM)

    # 2. Modello RNN (LSTM Seq2Seq)
    rnn_model = build_rnn_model(input_shape, OUTPUT_SEQ_LEN, FEATURE_DIM)
    rnn_history, rnn_prediction = train_and_evaluate(rnn_model, X, Y, "RNN (LSTM Seq2Seq)")

    # 3. Modello Transformer
    transformer_model = build_transformer_model(input_shape, OUTPUT_SEQ_LEN, FEATURE_DIM)
    transformer_history, transformer_prediction = train_and_evaluate(transformer_model, X, Y, "Transformer (Encoder Semplificato)")


    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    # 4. Visualizzazione dei risultati
    for i in range(FEATURE_DIM):
        generate_all_plots(X, Y, transformer_prediction, rnn_prediction, sample_index=0, feature_index=i)

    print(f"\nGenerazione di {FEATURE_DIM} grafici completata nella cartella 'predictions'.")


if __name__ == '__main__':
    # Disabilita gli avvisi di TensorFlow per un output più pulito
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    main()
