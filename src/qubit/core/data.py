import numpy as np
import pandas as pd
import os
from pathlib import Path

# --- Parametri del Dataset ---
N = 10  # Numero di Qubit
N_points = 1001  # Numero di passi temporali per traiettoria
N_traj = 400  # Numero di traiettorie
FEATURE_DIM = N + int(N * (N - 1) / 2)  # 10 Magnetizzazioni + 45 Correlazioni = 55
TOTAL_COLUMNS = 1 + FEATURE_DIM  # 1 (Tempo) + 55 (Feature) = 56

# --- Parametri di Previsione ---
INPUT_SEQ_LEN = 100  # Lunghezza della sequenza di input (Breve Termine)
OUTPUT_SEQ_LEN = 900  # Lunghezza della sequenza di output (Lungo Termine)


def load_and_process_data(file_path=None, traj_fraction=0.025):
    if file_path is None:
        # data.py -> core -> qubit -> src -> project_root
        project_root = Path(__file__).resolve().parents[3]
        file_path = project_root / "data" / "trajectories.csv"
    else:
        file_path = Path(file_path)

    """
    Carica il file CSV, lo pulisce e lo trasforma in array numpy
    strutturati per l'addestramento Seq2Seq.
    """
    if not os.path.exists(file_path):
        print(f"Errore: File dati non trovato a {file_path}.")
        return None, None

    print(f"Caricamento del file {file_path}...")
    # Carica il file CSV. Non ci sono header.
    df = pd.read_csv(file_path, header=None)

    # Verifica la dimensione del DataFrame
    if df.shape != (N_points * N_traj, TOTAL_COLUMNS):
        print(
            f"AVVISO: La dimensione del DataFrame ({df.shape}) non corrisponde all'atteso ({(N_points * N_traj, TOTAL_COLUMNS)}).")

    # Rimuove la colonna del tempo (la prima colonna)
    # Manteniamo solo le 55 feature (Magnetizzazioni + Correlazioni)
    data_features = df.iloc[:, 1:].values

    # Rimodella i dati in un array 3D: (N_traj, N_points, FEATURE_DIM)
    # Ogni traiettoria è una sequenza di N_points passi temporali
    # Usa solo una frazione delle traiettorie per un addestramento più rapido
    num_traj_to_use = int(N_traj * traj_fraction)
    data_features = data_features[:num_traj_to_use * N_points]
    data_3d = data_features.reshape(num_traj_to_use, N_points, FEATURE_DIM)

    print(f"Dati caricati e rimodellati in forma (Traiettorie, Passi Temporali, Feature): {data_3d.shape}")

    # --- Suddivisione in Input (X) e Output (Y) ---

    # X: Input (Breve Termine) - I primi INPUT_SEQ_LEN passi
    X = data_3d[:, :INPUT_SEQ_LEN, :]

    # Y: Output (Lungo Termine) - I successivi OUTPUT_SEQ_LEN passi
    # Nota: N_points = INPUT_SEQ_LEN + OUTPUT_SEQ_LEN (1001 = 100 + 901)
    # Se N_points = 1001, e INPUT_SEQ_LEN = 100, allora OUTPUT_SEQ_LEN dovrebbe essere 901.
    # Correggiamo la logica di slicing per usare esattamente 901 punti per l'output.
    # L'utente ha richiesto 900, ma 1001 - 100 = 901. Useremo 901 per l'output.

    # Se l'utente vuole esattamente 900, dobbiamo tagliare l'ultimo punto.
    # Per coerenza con N_points=1001 e Input=100, usiamo i restanti 901 punti.
    # Se l'utente vuole 900, lo faremo.

    # Calcola la lunghezza effettiva dell'output
    OUTPUT_SEQ_LEN_ACTUAL = N_points - INPUT_SEQ_LEN

    if OUTPUT_SEQ_LEN_ACTUAL != OUTPUT_SEQ_LEN:
        print(
            f"AVVISO: La lunghezza effettiva dell'Output è {OUTPUT_SEQ_LEN_ACTUAL} (1001 - 100), non {OUTPUT_SEQ_LEN}.")
        print(f"Verranno utilizzati {OUTPUT_SEQ_LEN_ACTUAL} passi per l'Output.")

    # X: Input (Breve Termine) - I primi INPUT_SEQ_LEN passi
    X = data_3d[:, :INPUT_SEQ_LEN, :]

    # Y: Output (Lungo Termine) - I successivi OUTPUT_SEQ_LEN_ACTUAL passi
    Y = data_3d[:, INPUT_SEQ_LEN:N_points, :]

    print(f"Forma dei dati di Input (X): {X.shape}")
    print(f"Forma dei dati di Output (Y): {Y.shape}")

    # Salvataggio dei dati pre-elaborati
    # np.savez('processed_quantum_data.npz', X=X, Y=Y)
    # print("Dati pre-elaborati salvati in processed_quantum_data.npz")

    return X, Y


if __name__ == '__main__':
    load_and_process_data(file_path='trajectories.csv')  # Il file è nella directory corrente