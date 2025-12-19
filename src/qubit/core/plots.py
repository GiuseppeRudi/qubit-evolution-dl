'''
import os
import numpy as np
import matplotlib.pyplot as plt


from ..model.dataset_splits import DatasetSplits

def _pick_pred(pred, sample_index: int):
    """
    Normalizza la predizione a shape (out_len, feature_dim) per un sample specifico.
    Supporta:
      - (n_samples, out_len, feature_dim)
      - (out_len, feature_dim)
    """
    pred = np.asarray(pred)
    if pred.ndim == 3:
        return pred[sample_index]
    if pred.ndim == 2:
        return pred
    raise ValueError(f"Shape predizione non supportata: {pred.shape}")


def generate_plot_for_feature(
    splits: DatasetSplits,
    transformer_prediction,
    rnn_prediction,
    sample_index: int = 0,
    feature_index: int = 0,
    out_dir: str = "predictions",
) -> str:
    """
    Genera un grafico (1 feature) confrontando:
    - dinamica reale (input + output)
    - previsione RNN
    - previsione Transformer

    Ritorna il path del file salvato.
    """
    X_test = splits.X_test
    Y_test = splits.Y_test

    if sample_index < 0 or sample_index >= X_test.shape[0]:
        raise IndexError(f"sample_index fuori range: {sample_index} (max {X_test.shape[0]-1})")

    if feature_index < 0 or feature_index >= X_test.shape[2]:
        raise IndexError(f"feature_index fuori range: {feature_index} (max {X_test.shape[2]-1})")

    # 1) sequenze reali
    input_sequence = X_test[sample_index, :, feature_index]   # (input_len,)
    true_output = Y_test[sample_index, :, feature_index]      # (out_len,)

    full_true = np.concatenate([input_sequence, true_output])
    input_len = len(input_sequence)
    full_len = len(full_true)

    # 2) predizioni (stesso sample)
    rnn_pred_2d = _pick_pred(rnn_prediction, sample_index)              # (out_len, feature_dim)
    tr_pred_2d = _pick_pred(transformer_prediction, sample_index)       # (out_len, feature_dim)

    rnn_output = rnn_pred_2d[:, feature_index]
    transformer_output = tr_pred_2d[:, feature_index]

    # allinea la lunghezza output se differisce da Y_test
    out_len = len(true_output)
    rnn_output = rnn_output[:out_len]
    transformer_output = transformer_output[:out_len]

    # 3) asse temporale
    time_axis = np.arange(full_len)
    pred_time_axis = np.arange(input_len, input_len + out_len)

    # 4) plot
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(15, 6))

    plt.plot(time_axis, full_true, label="Dinamica Reale (Ground Truth)", linewidth=2)
    plt.plot(pred_time_axis, rnn_output, label="Previsione RNN (LSTM)", linestyle="--")
    plt.plot(pred_time_axis, transformer_output, label="Previsione Transformer", linestyle="--")

    plt.axvline(x=input_len - 1, linestyle=":", label="Fine Input")

    # nome feature
    if feature_index < 10:
        feature_name = f"Magnetizzazione_m{feature_index + 1}"
    else:
        feature_name = f"Correlazione_c{feature_index - 9}"

    plt.title(f"Confronto Previsione Dinamica Quantistica: {feature_name}")
    plt.xlabel(f"Passi Temporali (Input: 0-{input_len-1}, Output: {input_len}-{input_len+out_len-1})")
    plt.ylabel("Valore della Feature")
    plt.legend()
    plt.grid(True)

    filename = f"sample_{sample_index}_feature_{feature_index+1}_{feature_name}.png"
    plot_path = os.path.join(out_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def generate_all_plots(
    splits: DatasetSplits,
    transformer_prediction,
    rnn_prediction,
    sample_index: int = 0,
    out_dir: str = "predictions",
):
    paths = []
    feature_dim = splits.X_test.shape[2]
    for feature_index in range(feature_dim):
        p = generate_plot_for_feature(
            splits, transformer_prediction, rnn_prediction,
            sample_index=sample_index,
            feature_index=feature_index,
            out_dir=out_dir
        )
        paths.append(p)
    return paths


''' 