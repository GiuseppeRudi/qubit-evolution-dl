import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

from numpy import random

from src.qubit.core.data import load_or_prepare_dataset
from src.qubit.rnn.models import build_rnn_model
from src.qubit.transformer.models import build_transformer_model  
from src.qubit.core.seed import set_seed
from src.qubit.core.utils import get_device
from src.qubit.core.plots import generate_all_plots

# --- Parametri di configurazione (corrispondenti al dataset reale) ---
INPUT_SEQ_LEN = 100
OUTPUT_SEQ_LEN = 901 # 1001 - 100 = 901
FEATURE_DIM = 55 # 10 Magnetizzazioni + 45 Correlazioni
EPOCHS = 100 # Aumentato per un addestramento più significativo
BATCH_SIZE = 32


def train_and_evaluate(model, splits, model_name):
    """Addestra e valuta un modello."""
    print(f"\n--- Addestramento del Modello: {model_name} ---")
    
    # Addestramento simulato
    history = model.fit(
        splits.X_train, 
        splits.Y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(splits.X_val, splits.Y_val),
    )
    
    print(f"Addestramento completato. Perdita finale (MSE): {history.history['loss'][-1]:.4f}")
    
    # Simulazione della previsione
    sample_input = splits.X_train[0:1]
    print(f"Sample Input: {sample_input.tolist()}")
    prediction = model.predict(sample_input)
    
    print(f"Previsione simulata per un campione di input:")
    print(f"  Forma dell'Input (Breve Termine): {sample_input.shape}")
    print(f"  Forma della Previsione (Lungo Termine): {prediction.shape}")
    print(f"  Previsione (primi 5 passi temporali):")
    np.set_printoptions(suppress=True, precision=16)
    print(prediction[0, :5, :])
    
    return history, prediction


def main():

    get_device()

    splits = load_or_prepare_dataset("src/qubit/core/config/dataset_base.yaml")

    input_shape = (INPUT_SEQ_LEN, FEATURE_DIM)

    # 2. Modello RNN (LSTM Seq2Seq)
    rnn_model = build_rnn_model(splits.X_train, splits.Y_train, latent_dim=64)
    rnn_history, rnn_prediction = train_and_evaluate(rnn_model, splits, "RNN (LSTM Seq2Seq)")

    # 3. Modello Transformer
    transformer_model = build_transformer_model(splits.X_train, splits.Y_train)
    transformer_history, transformer_prediction = train_and_evaluate(transformer_model, splits, "Transformer (Encoder Semplificato)")


    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    generate_all_plots(splits, transformer_prediction, rnn_prediction, sample_index=0)

    print(f"\nGenerazione di {FEATURE_DIM} grafici completata nella cartella 'predictions'.")


if __name__ == '__main__':
    main()
