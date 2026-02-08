
# Qubit Evolution DL

Qubit Evolution  is a research-oriented project that uses deep learning (LSTM and Transformer architectures) to model the **time evolution of multi-qubit systems** as a **sequence modeling** problem.

Given a past window of a quantum trajectory, the models can **predict future dynamics (Forecasting)** or **reconstruct missing timesteps (Super-Resolution)**.

---

## What this repository provides

- A complete **training pipeline** driven by a **single YAML configuration file**.
- Two model types:
    - **LSTM** baselines (Full-Seq, Step-Wise, Hybrid decoding)
    - **Transformer (TRN)** models (Hybrid encoder–decoder and Super-Resolution variant)
- A consistent **inference layer** via *adapters* (teacher forcing and free-running) to evaluate models in a comparable way.
- Tools for:
    - dataset split + windowing (forecasting and super-resolution)
    - saving training artifacts (predictions, splits, metadata)
    - generating plots (predictions + attention maps)

---

## Project structure

- `main.py` — project start point (loads YAML, builds model, trains, saves artifacts)
- `configs/` — experiment configuration files (LSTM/TRN, forecasting/SR, debug)
- `src/qubit/` — core package (data, models, strategies, training, inference, callbacks)
- `scripts/` — post-run plotting utilities:
    - `plot_from_data.py` (prediction plots from saved artifacts)
    - `plot_attention_maps.py` (attention heatmaps from saved attention maps)
- `tuning/` — hyperparameter tuning pipeline with Optuna
- `docs/` — documentation (setup, configuration, tuning, plotting)
- `data/trajectories.csv` — dataset

---

## Quick start

### 1) Local installation (Conda)

```bash
condaenv create -f environment.yml
conda activate qubit

# optional: reduce TensorFlow logs
condaenv config varsset TF_CPP_MIN_LOG_LEVEL=3
condaenv config varsset ABSL_LOGGING_LEVEL=ERROR

python install_tf.py
```

### 2) Run an experiment

Choose a YAML file  from `configs/` and run:

```bash
python main.py --run-cfg trn_hybrid
```

Switch model/variant/training by changing the config file name (e.g. `lstm_step_wise`, `trn_sr`, etc.).

---

## Outputs & artifacts

After training, the program creates a run directory that contains (depending on the config):

- `model.keras` (or equivalent) — saved model
- `data_splits.npz` — saved test split (and optionally mean/std for inverse standardization)
- `predictions.npz` — predictions
- `meta.json` — run metadata (feature names, shapes, config recap, etc.)
- `loss_plots/` — curve losses (if enabled)
- `attn_maps.npz` + attention plots (if enabled for Transformer runs)

The exact folder name includes the model type/variant/decoder_mode/experiment_name and a timestamp to keep runs isolated and reproducible.

---

## Where to read next

This README is intentionally short. Detailed usage is split into focused documents:

- `docs/setup.md`
    
    Installation instructions and environment notes (local + Docker).
    
- `docs/example.yaml`
    
    Full explanation of the main **experiment YAML**: dataset, windowing, model parameters, training phases, inference, and saving artifacts.
    
- `docs/tuning.md`
    
    How to run Optuna and how the dedicated tuning YAML (`configs/optuna.yaml`) controls the search space and scoring.
    
- `docs/plotting.md`
    
    How to use the plotting scripts:
    
    - prediction plots (`scripts/plot_from_data.py`)
    - attention heatmaps (`scripts/plot_attention_maps.py`)

- `reports/` — final PDF report and slides

---

## Docker (optional)

If you prefer Docker:

```bash
# RTX 50 image
docker build -t tf-app-rtx50 --build-arg TF_VARIANT=rtx50 .
docker run --gpus all -it tf-app-rtx50
```

or:

```bash
# Official TF image
docker build -t tf-app-official --build-arg TF_VARIANT=official .
docker run --gpus all -it tf-app-official
```

##